# 推論教學
在運行main.py前
請參考 pip list.txt 中的套件清單進行安裝

pip list.txt：列出專案所需的 Python 套件清單。

main.py：主程式，負責載入模型並執行推理。 使用之模型為Huggingface Model


# Datasets
datasets/：放置要使用的的資料集。

# Models
models/：儲存轉換後的模型檔案。

資較夾架構約為

![image](https://github.com/user-attachments/assets/68fa7803-170f-4021-acda-2a0a5e0de7d3)



# 原始權重轉為 Huggingface 格式
llama model to hf.txt：說明如何將 LLaMA 模型原始權重轉換為 Hugging Face 格式。供transformers 套件使用。

# flash_attention 在 Windows 使用教學
flash_attention in Windows.txt：在 Windows 環境中使用 Flash Attention 的設定說明。

# Huggingface 格式 轉為 GGUF(GPT-Generated Unified Format)
hf_to_GGUF.txt：指導如何將 Hugging Face 模型轉換為 GGUF 格式。 供AnythingLLM 或是 LM studio 等應用程式使用。

# GGUF 本地 model 在 LMstudio 使用教學
安裝好後會在以下介面

![image](https://github.com/user-attachments/assets/62208e43-81fa-4ec8-bf7f-4d4786935d45)

點選我的模型

![image](https://github.com/user-attachments/assets/d6ad9180-5254-4a25-b292-4520f6ed4689)

在點模型目錄旁邊的... 在檔案總管打開

![image](https://github.com/user-attachments/assets/4a32e9f6-ef95-4c0f-a429-d81901847f96)

![image](https://github.com/user-attachments/assets/d45c6844-c57d-4ab5-a757-ad9b8153a433)

新增一個資料夾叫 lmstudio-community

![image](https://github.com/user-attachments/assets/fb528624-934d-442b-a882-5c5c23c53d80)

再新增一個資料夾，將 GGUF 檔案放進去(副檔名要是.gguf)

![image](https://github.com/user-attachments/assets/25560f3f-663b-48e4-9934-a7fcbe326b2a)

就可以使用 本地model 進行推論了

![image](https://github.com/user-attachments/assets/aec2a67c-c759-46e0-a217-76773e5ad8c2)

開始使用前記得先點進

![image](https://github.com/user-attachments/assets/824f010a-01e1-476e-8385-bd15923f3f26)

提示詞模板改為 你所使用的 model

![image](https://github.com/user-attachments/assets/b9133d82-9f80-424d-bbae-6b4ab47882b8)

