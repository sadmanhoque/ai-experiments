from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from bs4 import BeautifulSoup
import torch
import os
import glob

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")


def load_html_files(html_folder_path):
    """
    Load all HTML files from a folder and extract clean text
    """
    html_files = glob.glob(os.path.join(html_folder_path, "*.html"))
    documents = []

    for file_path in html_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get clean text
            text = soup.get_text(separator=' ', strip=True)
            # Clean up whitespace
            text = ' '.join(text.split())

            documents.append({
                'filename': os.path.basename(file_path),
                'text': text
            })

    return documents


def create_qa_dataset(documents, qa_pairs):
    """
    Create a dataset from documents and Q&A pairs

    qa_pairs format:
    [
        {
            'filename': 'page1.html',
            'question': 'What is the capital of France?',
            'answer': 'Paris',
            'answer_start': 150  # character position in the cleaned text
        },
        ...
    ]
    """
    data = {'context': [], 'question': [], 'answers': []}

    # Create a lookup dict for documents
    doc_dict = {doc['filename']: doc['text'] for doc in documents}

    for qa in qa_pairs:
        context = doc_dict.get(qa['filename'])
        if context:
            data['context'].append(context)
            data['question'].append(qa['question'])
            data['answers'].append({
                'text': [qa['answer']],
                'answer_start': [qa['answer_start']]
            })

    return Dataset.from_dict(data)


# ==== CONFIGURE YOUR DATA HERE ====

# Path to folder containing your HTML files
HTML_FOLDER = "./test-files"

# Your question-answer pairs
# You need to create these for your HTML pages
qa_pairs = [
    {
        'filename': 'example.html',
        'question': 'What is discussed in this page?',
        'answer': 'example answer',
        'answer_start': 0  # position where answer starts in cleaned text
    },
    # Add more Q&A pairs here
]

# ==== LOAD AND PREPARE DATA ====

print("Loading HTML files...")
documents = load_html_files(HTML_FOLDER)
print(f"Loaded {len(documents)} HTML files")

print("Creating dataset...")
dataset = create_qa_dataset(documents, qa_pairs)
print(f"Created {len(dataset)} training examples")


# Tokenize the dataset
def tokenize_qa(examples):
    tokenized = tokenizer(
        examples['question'],
        examples['context'],
        truncation=True,
        padding='max_length',
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True
    )

    offset_mapping = tokenized.pop("offset_mapping")
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answer = examples['answers'][sample_idx]
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])

        token_start_index = 0
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        token_end_index = len(offsets) - 1
        while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    return tokenized


if len(dataset) > 0:
    tokenized_dataset = dataset.map(tokenize_qa, batched=True, remove_columns=dataset.column_names)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./custom_html_qa_model",
        evaluation_strategy="no",  # Change to "epoch" if you have validation data
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=3e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print("\nSetup complete!")
    print("To train: uncomment trainer.train() below")
    print("Don't forget to add your HTML files and Q&A pairs!")

    # Uncomment to start training
    # trainer.train()
    # trainer.save_model("./custom_html_qa_final_model")
else:
    print("\nNo training data created. Please add HTML files and Q&A pairs.")

#For testing individual actions/functions
'''
if __name__ == '__main__':
    print(load_html_files(HTML_FOLDER))
'''