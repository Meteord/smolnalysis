# **Hackathon Project: OpenDataAgent & Intelligent Data Analysis App**

Task tracker: [task_list.md](task_list.md)

---

## **🎯 Vision**

### **Minimal Solution (MVP)**

An **interactive Gradio app** that:

- Allows **CSV file uploads** for analysis.
- Uses **two specialized language models:**
  - **Data Analysis Model:** Automated data evaluation and visualization.
  - **OpenUI-Lang Translation Model:** Converts natural language into OpenUI commands.

### **Extended Solution (Stretch Goal)**

**OpenDataAgent** – A modular system featuring:

- **Model Zoo:**
  - **CKAN Model:** Integration and querying of open datasets (e.g., from portals like [offenedata.de](https://offenedata.de)).
  - **OpenUI Model:** Specialized in translating user queries into OpenUI syntax.
  - **Model Router:** Dynamic selection of the best model for the task at hand.

---

## **🔧 Technical Implementation**

### **1. Gradio App**

- **Frontend:** User-friendly interface for uploading and analyzing CSV files.
- **Backend:** Integration of language models for real-time analysis.
- **Extension:** Support for **OpenUI commands** (e.g., `/plot histogram of column X`).

### **2. Model Fine-Tuning**

- **Goal:** Adapt models to **OpenUI-Lang output** for precise and user-friendly results.
- **Data:** Generate training data for:
  - Data analysis queries (e.g., "Show me the correlation between column A and B").
  - OpenUI translations (e.g., "Create a bar chart" → `/plot bar chart`).

### **3. Model Zoo**

- **CKAN Integration:** Automated querying and processing of open datasets.
- **OpenUI Specialization:** Optimization for generating UI commands.
- **Router Training:** Develop a system to select the appropriate model based on the query.



## **💡 Why This Project?**

- **Innovation:** Combines **data analysis** with **language-driven UI control**. 
- **Practical Use:** Simplifies working with open data and visualizations.
- **Scalable:** Model Zoo can be extended with additional specializations.
- **Community Contribution:** Open-source potential for developers and data enthusiasts.