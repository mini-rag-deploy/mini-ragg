from fastapi import FastAPI
from routes import base, data, nlp
from helpers.config import Settings
from stores.llm import LLMProviderFactory
from stores.llm.templates.template_parser import template_parser
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

app = FastAPI()


async def startup_span():
    settings = Settings()

    postgres_conn=f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    app.db_engine = create_async_engine(postgres_conn)
    app.db_client = sessionmaker(
                    app.db_engine, expire_on_commit=False,class_=AsyncSession
                                )

    llm_provider_factory = LLMProviderFactory(settings)
    vector_db_provider_factory = VectorDBProviderFactory(settings,db_client=app.db_client)

    # generation client
    app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    app.generation_client.set_generation_mode(model_id=settings.GENERATION_MODEL_ID)

    # embedding client
    app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE
    )

    # vector database client
    app.vectordb_client = vector_db_provider_factory.create(provider=settings.VECTORD_DB_BACKEND)
    await app.vectordb_client.connect()

    app.template_parser = template_parser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG
    )

async def shutdown_span():
    app.db_engine.dispose()
    await app.vectordb_client.disconnect()

app.on_event("startup")(startup_span)
app.on_event("shutdown")(shutdown_span)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)


 











# @app.get("/welcome")
# async def welcome():

    

#     return {
#         "message": "Welcome to the FastAPI application!"
#     }
