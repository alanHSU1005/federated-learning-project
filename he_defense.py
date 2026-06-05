# ==============================================================================
# he_defense.py — 聯邦學習同態加密防禦模組 (Step 3-1 Bonus)
# 
# 本模組採用微軟 SEAL 底層優化的 TenSEAL 庫，實現 CKKS 加密方案。
# 核心邏輯：
#   1. 客戶端：將訓練後的權重張量拉平，並利用 SIMD 打包加密為密文向量 (Ciphertext Vector)
#   2. 伺服器：在完全不解密的情況下，直接對各客戶端的密文進行加權聚合 (Homomorphic FedAvg)
#   3. 評估端：將聚合後的全局密文解密，還原為 PyTorch Tensor 載入全局模型以利評估 Accuracy
# ==============================================================================

import time
import numpy as np
import torch
import tenseal as ts
from typing import List, Dict

import config

def create_tenseal_context(poly_modulus_degree: int = 8192) -> ts.Context:
    """
    建立 TenSEAL CKKS 同態加密全域環境。
    
    安全規格論述 (Privacy Level)：
      - 當 degree = 8192 時，總模數位數上限為 119 bits 即可達到真實的 256-bit 安全強度。
      - 本專案採用 [40, 30, 40] 配置（總位數 110 bits），嚴格符合 256-bit 安全標準。
      - 全域縮放因子 (Scale) 設為 2^30，確保在 LeNet-5 的深度下具有極佳的小數精準度。
    """
    if poly_modulus_degree == 8192:
        coeff_mod_bit_sizes = [40, 30, 40]  # 總計 110 bits，嚴格符合 256-bit 安全標準
        global_scale = 2 ** 30
    elif poly_modulus_degree == 4096:
        coeff_mod_bit_sizes = [30, 20, 30]
        global_scale = 2 ** 20
    else:
        raise ValueError("不支援的 poly_modulus_degree！請選擇 4096 或 8192。")

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes
    )
    context.global_scale = global_scale
    context.generate_galois_keys()
    return context


class HEDefenseAggregator:
    """
    🛡️ 同態加密安全聚合處理器 (HEDefenseAggregator)
    """
    def __init__(self, poly_modulus_degree: int = 8192, server_model=None, **kwargs):
        self.poly_modulus_degree = poly_modulus_degree
        self.slots = self.poly_modulus_degree // 2
        self.server_model = server_model

        # ─── 🔑 實現真實世界私鑰隔離 (Secret Key Isolation) ───
        # 1. 客戶端角色：持有完整金鑰環境（含公鑰、私鑰）
        self.client_context = create_tenseal_context(self.poly_modulus_degree)

        # 2. 伺服器角色：序列化時不保存私鑰，再反序列化為純公鑰環境
        #    make_context_public() 後，server_context 真正無法解密任何密文
        self.server_context = ts.context_from(
            self.client_context.serialize(save_secret_key=False)
        )
        # 雙重確認：確保 server_context 確實不含私鑰
        assert not self.server_context.has_secret_key(), \
            "[HE] 安全警告：伺服器 context 意外含有私鑰，請檢查初始化流程！"

        print(f"   - 加密方案: CKKS (適用於深度學習浮點數)")
        print(f"   - 隱私等級 (Key Size): {poly_modulus_degree} bits")

    def aggregate(self, client_updates: List[Dict], **kwargs) -> Dict:
        """
        對所有客戶端上傳的明文權重進行「密文盲算聚合」與「自動精度誤差量化」。
        """
        if not client_updates:
            return {}

        # 計算總樣本數與各客戶端權重比例
        total_samples = sum([u['num_samples'] for u in client_updates])
        weight_factors = [u['num_samples'] / total_samples for u in client_updates]

        sample_state_dict = client_updates[0]['weights']
        aggregated_state_dict = {}

        # ─── 🔍 核心亮點：預先一次性計算所有層的明文 FedAvg 基準線 ───
        # 移至最外層迴圈，避免在每個 layer 內重複遍歷所有客戶端，消除效能冗餘。
        plaintext_reference = {}
        for key in sample_state_dict.keys():
            if isinstance(sample_state_dict[key], torch.Tensor):
                layer_avg = client_updates[0]['weights'][key].detach().cpu().float() * weight_factors[0]
                for client_idx in range(1, len(client_updates)):
                    layer_avg = layer_avg + client_updates[client_idx]['weights'][key].detach().cpu().float() * weight_factors[client_idx]
                plaintext_reference[key] = layer_avg
        # ─────────────────────────────────────────────────────────────

        # 耗時量測計時器
        client_enc_time = 0.0
        server_agg_time = 0.0
        dec_load_time = 0.0

        print(f"⚙️  正在執行 {len(sample_state_dict)} 個模型層的安全密文盲算...")

        for key in sample_state_dict.keys():
            if not isinstance(sample_state_dict[key], torch.Tensor):
                aggregated_state_dict[key] = sample_state_dict[key]
                continue

            layer_shape = sample_state_dict[key].shape
            orig_numel = sample_state_dict[key].numel()

            # ------------------------------------------------------------------
            # 【階段 1】客戶端本地端：展平模型、分塊打包並進行「同態加密」
            #           加密後立即序列化（模擬網路傳輸），伺服器端只收到位元組
            # ------------------------------------------------------------------
            t_enc_start = time.time()
            # 每個客戶端的密文以序列化位元組串列傳遞，伺服器無從直接存取密文物件
            client_serialized_chunks: List[List[bytes]] = []

            for u in client_updates:
                flat_weight = u['weights'][key].detach().cpu().numpy().flatten().astype(np.float64)

                chunks = [flat_weight[i:i + self.slots] for i in range(0, orig_numel, self.slots)]
                serialized_chunks: List[bytes] = []

                for chunk in chunks:
                    if len(chunk) < self.slots:
                        padded_chunk = np.pad(chunk, (0, self.slots - len(chunk)), 'constant')
                    else:
                        padded_chunk = chunk

                    # 🔒 客戶端：使用含私鑰的 client_context 加密
                    enc_vec = ts.ckks_vector(self.client_context, padded_chunk.tolist())
                    # 🌐 序列化：模擬將密文透過網路傳送給伺服器（此後私鑰不再可達）
                    serialized_chunks.append(enc_vec.serialize())

                client_serialized_chunks.append(serialized_chunks)

            client_enc_time += (time.time() - t_enc_start)

            # ------------------------------------------------------------------
            # 【階段 2】中央伺服器端：完全密文狀態下的「同態加權聚合」
            #           使用無私鑰的 server_context 反序列化，確保伺服器真正無法解密
            # ------------------------------------------------------------------
            t_agg_start = time.time()
            num_chunks = len(client_serialized_chunks[0])
            server_aggregated_chunks: List[ts.CKKSVector] = []

            for chunk_idx in range(num_chunks):
                # 🏢 伺服器端：從位元組流反序列化為密文，使用純公鑰 server_context
                #    server_context 不含私鑰，此步驟在技術上無法解密
                first_enc = ts.ckks_vector_from(
                    self.server_context,
                    client_serialized_chunks[0][chunk_idx]
                )
                agg_chunk = first_enc * weight_factors[0]

                for client_idx in range(1, len(client_updates)):
                    other_enc = ts.ckks_vector_from(
                        self.server_context,
                        client_serialized_chunks[client_idx][chunk_idx]
                    )
                    agg_chunk += other_enc * weight_factors[client_idx]

                server_aggregated_chunks.append(agg_chunk)

            server_agg_time += (time.time() - t_agg_start)

            # ------------------------------------------------------------------
            # 【階段 3】安全解密端：密文還原、對齊裁切與張量重塑
            #           模擬密文傳回客戶端，使用 client_context 私鑰解密
            # ------------------------------------------------------------------
            t_dec_start = time.time()
            flat_decrypted = []

            for agg_chunk in server_aggregated_chunks:
                # 🔓 客戶端：使用含私鑰的 client_context 解密聚合後密文
                decrypted_chunk = agg_chunk.decrypt(self.client_context.secret_key())
                flat_decrypted.extend(decrypted_chunk)

            # ✂️ 裁切對齊機制：精確切除末端補上的零
            flat_decrypted = np.array(flat_decrypted[:orig_numel], dtype=np.float32)

            recon_tensor = torch.from_numpy(flat_decrypted).reshape(layer_shape).to(config.DEVICE)
            aggregated_state_dict[key] = recon_tensor

            dec_load_time += (time.time() - t_dec_start)

            # ------------------------------------------------------------------
            # 🔍 自動解密誤差精確量化與列印
            # ------------------------------------------------------------------
            if key in plaintext_reference:
                with torch.no_grad():
                    max_error = (recon_tensor.cpu() - plaintext_reference[key]).abs().max().item()
                    print(f"   [🔍 CKKS 誤差檢查] 層 {key:<30} 最大解密誤差: {max_error:.8e}")

        # 計算總開銷
        total_he_overhead = client_enc_time + server_agg_time + dec_load_time

        print(f"\n⏱️  [HE Performance Metrics - 通訊輪次耗時度量]:")
        print(f"     - 客戶端總加密耗時 (Client Encryption Time)    : {client_enc_time:.4f} 秒")
        print(f"     - 伺服器同態聚合耗時 (Server Aggregation Time)  : {server_agg_time:.4f} 秒")
        print(f"     - 系統解密與載入耗時 (Decryption & Load Time)   : {dec_load_time:.4f} 秒")
        print(f"     - 🚀 當前同態防禦總開銷 (Total HE Overhead)      : {total_he_overhead:.4f} 秒\n")

        if self.server_model is not None:
            from model import set_model_weights
            set_model_weights(self.server_model, aggregated_state_dict)

        return aggregated_state_dict