# Chat with PDFs: RAG with LangChain, GPT & LLama in Python

How to use LangChain and LLMs to do RAG in Python and ask questions about PDF documents.

## Packages

- OpenAI
- LangChain
- LangChain Community
- Faiss

## Environment Setup

- pyenv virtualenv --version
- pyenv virtualenvs
- pyenv virtualenv chat-with-pdfs
- pyenv activate chat-with-pdfs --verbose
- pip install -r requirements.txt

## How it works

- OpenAI: please add `OPENAI_API_KEY` to `.env` first

```
python src/openai_main.py
```

- Ollama: pls install ollama first

```
python src/ollama_main.pry
```

## References

- [Ollama Setup](https://ollama.com/)
- [Langchain OpenAI](https://python.langchain.com/docs/integrations/providers/openai/)
- [Langchain Ollama](https://python.langchain.com/docs/integrations/providers/ollama/)
