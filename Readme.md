# **Mathematical Word Problem Solving using NLP**

This project focuses on advancing mathematical reasoning capabilities in NLP models for solving Mathematical Word Problems (MWPs). By leveraging both traditional architectures and advanced language models, it evaluates their efficacy in bridging the semantic gap between natural language and mathematical logic.

---

## **Overview**

Mathematical Word Problems consist of natural language narratives that describe real-world scenarios and pose questions about unknown quantities. Solving these problems requires translating text into structured mathematical expressions, which remains a significant challenge for NLP models.

This project evaluates:
1. **Baseline models**: Feedforward networks, LSTMs, and Transformers.
2. **Math-specific models**: Goal-Driven Tree-Structured (GTS) and Graph-to-Tree models.
3. **Large Language Models (LLMs)**: Fine-tuned Gemma-2-9B and Mistral-Instruct-7B.

---

## **Key Features**

1. **Dataset:**
   - Trained on **MAWPS** and **ASDiv-A** datasets.
   - Evaluated on the **SVAMP** dataset to assess reasoning robustness and sensitivity to question phrasing.

2. **Model Approaches:**
   - **Baselines:** Seq2Seq models with variations of encoders (FFN, LSTM, Transformer).
   - **Task-Specific:** Graph-to-Tree models with graph-based encoding and reasoning.
   - **LLMs:** Advanced reasoning with techniques like Chain-of-Thought (CoT) prompting and Few-Shot fine-tuning.

3. **Performance Highlights:**
   - Baseline transformer models achieved up to **27.9% accuracy**.
   - GTS and Graph-to-Tree models showed structured reasoning improvements, achieving **36.1% accuracy**.
   - Fine-tuned LLMs like Gemma-2-9B, with CoT and Few-Shot prompting, achieved **> 90% accuracy**.

---


### **Dependencies**
- Python >= 3.6
- PyTorch >= 1.8
- Transformers >= 4.9.1


## **Results and Analysis**

- **Baseline Models:** Limited performance due to lack of structured reasoning capabilities.
- **Task-Specific Models:** GTS and Graph-to-Tree models excel at structured reasoning, demonstrating enhanced performance.
- **LLMs:** Significant performance gains with Few-Shot and CoT prompting, with Gemma-2-9B achieving a **> 90% accuracy** on SVAMP dataset.

---

## **Contributors**
- **Swayam Agrawal**
- **Pratham Thakkar**
- **Yash Kawade**

---

- Refer to the Report for greater details.
