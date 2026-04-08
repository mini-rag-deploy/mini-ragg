from fastapi import FastAPI, APIRouter, Depends, UploadFile, status , Request
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers import NLPController
import aiofiles
from models import ResponseSignal
import logging
from tqdm.auto import tqdm

from .schemes.nlp import PushRequest , SearchRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel
from models.db_schemes import DataChunk
from models.db_schemes import Asset
from models.enums.AssetTypeEnum import AssetTypeEnum
from tasks.data_indexing import index_data_content

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"],
)

@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: int, push_request: PushRequest,
                        app_settings: Settings = Depends(get_settings)):
    task = index_data_content.delay(project_id=project_id, do_reset=push_request.do_reset)

    return JSONResponse(
                        content={
                            "signal": ResponseSignal.DATA_PUSH_TASK_READY.value,
                            "task_id": task.id
                        }
    )


@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info(request: Request, project_id: int):
    project_model = await ProjectModel.create_instence(
        db_client=request.app.db_client,
    )
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"signal": ResponseSignal.PROJECT_NOT_FOUND.value})
    
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client
    )

    collection_info = await nlp_controller.get_vector_db_collection_info(project=project)

    if collection_info is None:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"signal": ResponseSignal.COLLECTION_NOT_FOUND.value})
    
    return JSONResponse(status_code=status.HTTP_200_OK,
                         content={"signal": ResponseSignal.COLLECTION_INFO_RETRIEVE_SUCCESS.value,
                                  "collection_info": collection_info})
@nlp_router.post("/index/search/{project_id}")
async def search_index(request: Request, project_id: int, search_request: SearchRequest):
    project_model = await ProjectModel.create_instence(
        db_client=request.app.db_client,
    )
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"signal": ResponseSignal.PROJECT_NOT_FOUND.value})
    
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )

    search_results = await nlp_controller.search_vector_db_collection(
        project=project,
        text=search_request.text,
        limit=search_request.limit
    )

    if not search_results:
        return JSONResponse(status_code=status.HTTP_200_OK, content={"signal": ResponseSignal.VECTOR_DB_SEARCH_ERROR.value, "results": []})
    
    return JSONResponse(status_code=status.HTTP_200_OK, content={"signal": ResponseSignal.VECTOR_DB_SEARCH_SUCCESS.value,
                                                                  "results": [result.dict() for result in search_results]})



@nlp_router.post("/index/answer/{project_id}")
async def answer_rag_question(request: Request, project_id: int, search_request: SearchRequest):
    project_model = await ProjectModel.create_instence(
        db_client=request.app.db_client,
    )
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"signal": ResponseSignal.PROJECT_NOT_FOUND.value})
    
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )

    answer,full_peompt, chat_history = await nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit
    )

    if not answer:
        return JSONResponse(status_code=status.HTTP_200_OK, content={"signal": ResponseSignal.RAG_ANSWERING_ERROR.value, "answer": None})
    
    return JSONResponse(status_code=status.HTTP_200_OK,
                         content={
                            "signal": ResponseSignal.RAG_ANSWERING_SUCCESS.value,
                            "answer": answer,
                            "full_prompt": full_peompt,
                            "chat_history": chat_history
                        }
    )
