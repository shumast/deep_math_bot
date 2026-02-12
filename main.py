import os
import asyncio
import nest_asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import langchain
langchain.debug = False

nest_asyncio.apply()

tg_token = 'CHANGE_ME'
hf_token = 'CHANGE_ME'
os.environ['HF_TOKEN'] = hf_token

def setup_rag_system(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()
    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct", 
        temperature=0.3,
        max_new_tokens=1024,
        repetition_penalty=1.1,
        huggingfacehub_api_token=hf_token,
    )
    llm = ChatHuggingFace(llm=endpoint)

    prompt_template = """Ты — полезный ассистент, который помогает находить ответы в документах. 
    Твоя задача: изучить предоставленный текст и ответить на вопрос пользователя максимально полно и понятно.

    Контекст:
    {context}

    Вопрос: {question}

    Правила ответа:
    1. Пиши развернуто и связно, как человек. Избегай обрывистых фраз.
    2. Если контекст на английском, переведи его на качественный русский язык.
    3. Если информации в тексте недостаточно, так и скажи. Не выдумывай факты.
    4. Структурируй ответ, если в нем несколько пунктов.
    5. Начинай ответ с фразы "Опираясь на тво"
    6. ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ

    Полезный ответ:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa_chain = RetrievalQA.from_llm(llm,
        retriever=retriever,
        return_source_documents=True,
        prompt=PROMPT,
        llm_chain_kwargs={
            "verbose": True
        })
    return qa_chain

qa_chain = None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Здравствуйте! Я бот, который может помочь вам работать с PDF файлами. '
                                    'Отправьте мне PDF файл, и я разберу его на текстовые фрагменты. '
                                    'Затем вы сможете задавать мне вопросы по содержимому файла, и я постараюсь на них ответить.')


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global qa_chain

    document = update.message.document
    if document and document.file_name.lower().endswith('.pdf'):
        file = await document.get_file()
        file_path = f'./{document.file_name}'
        await file.download_to_drive(file_path)
        await update.message.reply_text(f'Файл {document.file_name} сохранён. Начинаю парсинг...')

        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            parsed_text = "\n\n".join([doc.page_content for doc in documents])

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_text(parsed_text)

            await update.message.reply_text(f'Файл распаршен на {len(chunks)} чанков.')
            qa_chain = setup_rag_system(documents)
            await update.message.reply_text('Теперь вы можете задавать вопросы по содержимому файла.')
        except Exception as e:
            await update.message.reply_text(f'Произошла ошибка при парсинге файла: {str(e)}')
    else:
        await update.message.reply_text('Пожалуйста, отправьте файл в формате PDF.')


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global qa_chain

    if qa_chain:
        query = update.message.text
        try:
            response = qa_chain.invoke(query)
            result = response['result']
            result = result[result.find('Полезный ответ:') + len('Полезный ответ:\n'):].strip()
            await update.message.reply_text(result)
        except Exception as e:
            await update.message.reply_text(f'Произошла ошибка при обработке запроса: {str(e)}')
    else:
        await update.message.reply_text('Пожалуйста, сначала отправьте PDF файл для парсинга.')


async def main() -> None:
    application = Application.builder().token(tg_token).build()

    print('Бот запущен')

    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    await application.run_polling()

if __name__ == '__main__':
    asyncio.run(main())
