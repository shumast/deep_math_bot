import os
import json
import uuid
import asyncio
import nest_asyncio
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from config import tg_token, hf_token

nest_asyncio.apply()
os.environ["HF_TOKEN"] = hf_token


DATA_DIR = Path("data/users")
DATA_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDINGS_MODEL,
    model_kwargs={"device": "cpu"},
)


def get_user_dir(user_id: int) -> Path:
    path = DATA_DIR / str(user_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_files_list_path(user_id: int) -> Path:
    return get_user_dir(user_id) / "files.json"


def get_doc_ids_path(user_id: int) -> Path:
    return get_user_dir(user_id) / "doc_ids.json"


def load_user_files(user_id: int):
    path = get_files_list_path(user_id)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_user_files(user_id: int, files):
    with open(get_files_list_path(user_id), "w", encoding="utf-8") as f:
        json.dump(files, f, ensure_ascii=False, indent=2)


def load_doc_ids(user_id: int):
    path = get_doc_ids_path(user_id)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_doc_ids(user_id: int, data):
    with open(get_doc_ids_path(user_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_vectorstore(user_id: int):
    user_path = get_user_dir(user_id)
    if (user_path / "index.faiss").exists():
        return FAISS.load_local(
            folder_path=str(user_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    return None


def save_vectorstore(user_id: int, vectorstore: FAISS):
    user_path = get_user_dir(user_id)
    vectorstore.save_local(str(user_path))


def build_qa_chain(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    template = """
Ты — математический ассистент, который отвечает строго на основе контекста.

Контекст:
{context}

Вопрос: {question}

Требования к ответу:
1. Отвечай ТОЛЬКО на русском языке.
2. Используй только информацию из контекста.
3. Если информации недостаточно — напиши:
   "В загруженных документах нет полной информации по этому вопросу."
4. Ответ должен быть развернутым и логически завершённым.
5. Если упоминаются свойства, обязательно перечисли и раскрой каждое.
6. Используй структуру:
   - Краткое определение
   - Подробное объяснение
   - Перечень свойств (если применимо)
   - Дополнительные пояснения

Ответ:
"""

    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )

    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        temperature=0.1,
        max_new_tokens=1024,
        repetition_penalty=1.1,
        huggingfacehub_api_token=hf_token,
    )

    llm = ChatHuggingFace(llm=endpoint)

    return RetrievalQA.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        prompt=PROMPT,
        return_source_documents=True,
    )



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Здравствуйте!\n\n"
        "Я бот с персональной PDF-библиотекой 📚\n"
        "Загружайте PDF-файлы и задавайте вопросы."
    )


async def library(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    files = load_user_files(user_id)

    if not files:
        await update.message.reply_text("Ваша библиотека пуста.")
        return

    text = "📚 Ваша библиотека:\n\n"
    for i, file in enumerate(files, 1):
        text += f"{i}. {file}\n"

    await update.message.reply_text(text)


async def clear_library(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_path = get_user_dir(user_id)

    for file in user_path.glob("*"):
        file.unlink()

    await update.message.reply_text("Библиотека полностью очищена.")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    document = update.message.document

    if not document.file_name.lower().endswith(".pdf"):
        await update.message.reply_text("Пожалуйста, отправьте PDF файл.")
        return

    await update.message.reply_text("Файл получен. Обрабатываю...")

    user_path = get_user_dir(user_id)
    file_path = user_path / document.file_name

    file = await document.get_file()
    await file.download_to_drive(str(file_path))

    loader = PyPDFLoader(str(file_path))
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    docs = text_splitter.split_documents(documents)

    doc_ids = []
    for doc in docs:
        doc_id = str(uuid.uuid4())
        doc.metadata["source_file"] = document.file_name
        doc.metadata["doc_id"] = doc_id
        doc_ids.append(doc_id)

    vectorstore = load_vectorstore(user_id)

    if vectorstore:
        vectorstore.add_documents(docs, ids=doc_ids)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings, ids=doc_ids)

    save_vectorstore(user_id, vectorstore)

    files = load_user_files(user_id)
    if document.file_name not in files:
        files.append(document.file_name)
        save_user_files(user_id, files)

    doc_ids_map = load_doc_ids(user_id)
    doc_ids_map[document.file_name] = doc_ids
    save_doc_ids(user_id, doc_ids_map)

    await update.message.reply_text(
        f"Документ '{document.file_name}' добавлен в библиотеку."
    )


async def remove_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id

    files = load_user_files(user_id)
    doc_ids_map = load_doc_ids(user_id)

    if not files:
        await update.message.reply_text("Ваша библиотека пуста.")
        return

    if not context.args:
        await update.message.reply_text(
            "Использование:\n/remove НОМЕР_ФАЙЛА\n\n"
            "Посмотреть номера можно через /library"
        )
        return

    try:
        file_index = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Нужно указать номер файла (число).")
        return

    if file_index < 1 or file_index > len(files):
        await update.message.reply_text("Неверный номер файла.")
        return

    filename = files[file_index - 1]

    vectorstore = load_vectorstore(user_id)
    if not vectorstore:
        await update.message.reply_text("Индекс не найден.")
        return

    ids_to_delete = doc_ids_map.get(filename, [])

    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)
        save_vectorstore(user_id, vectorstore)

    file_path = get_user_dir(user_id) / filename
    if file_path.exists():
        file_path.unlink()

    files.pop(file_index - 1)
    save_user_files(user_id, files)

    doc_ids_map.pop(filename, None)
    save_doc_ids(user_id, doc_ids_map)

    await update.message.reply_text(
        f"Файл '{filename}' удалён."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    vectorstore = load_vectorstore(user_id)

    if not vectorstore:
        await update.message.reply_text(
            "Сначала загрузите хотя бы один PDF файл."
        )
        return

    query = update.message.text

    docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)

    if not docs_with_scores:
        await update.message.reply_text(
            "В загруженных документах нет информации по этому вопросу."
        )
        return

    best_score = docs_with_scores[0][1]

    if best_score > 1.0:
        await update.message.reply_text(
            "В загруженных документах нет информации по этому вопросу."
        )
        return

    qa_chain = build_qa_chain(vectorstore)

    await update.message.reply_text("Ищу ответ...")

    try:
        response = qa_chain.invoke(query)

        if not response.get("source_documents"):
            await update.message.reply_text(
                "В загруженных документах нет информации по этому вопросу."
            )
            return

        result = response["result"]

        if any('\u4e00' <= ch <= '\u9fff' for ch in result):
            await update.message.reply_text(
                "В загруженных документах нет информации по этому вопросу."
            )
            return

        await update.message.reply_text(result)

    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")




async def main():
    application = Application.builder().token(tg_token).build()

    print("Bot started")

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("library", library))
    application.add_handler(CommandHandler("clear", clear_library))
    application.add_handler(CommandHandler("remove", remove_file))

    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    await application.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
