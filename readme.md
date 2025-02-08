
# Write me project -  Information retrieval

A comprehensive NLP pipeline for analyzing email and chat communications using various techniques including Named Entity Recognition, Topic Modeling, and Dialogue Analysis. -

## Setup


2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the required datasets from Kaggle:
- [Clinton Emails Dataset](https://www.kaggle.com/datasets/kaggle/hillary-clinton-emails)
- [Enron Emails Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)
- [Fraudulent Emails Dataset](https://www.kaggle.com/datasets/rtatman/fraudulent-email-corpus)
- [NPS Chat Corpus](https://www.kaggle.com/datasets/nltkdata/nps-chat)

5. Create a `datasets` directory in the project root and organize the downloaded data as follows:

```bash
mkdir datasets
datasets/
├── clinton_emails/
│ └── clinton_emails.csv
├── enron_emails/
│ └── enron_emails.csv
├── fraudulent_emails/
│ └── fraudulent_emails.csv
└── nps_chat/
└── 10-19-20s.xml
└── other_xml_files.xml
```

6. Run the project:

- Open the notebook 
- Swap general_path with your main path
- Run the needed cells.


