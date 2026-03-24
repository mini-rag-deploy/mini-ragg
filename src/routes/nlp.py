from fastapi import FastAPI, APIRouter, Depends, UploadFile, status , Request
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers import NLPController
import aiofiles
from models import ResponseSignal
import logging

from .schemes.nlp import PushRequest , SearchRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel
from models.db_schemes import DataChunk
from models.db_schemes import Asset
from models.enums.AssetTypeEnum import AssetTypeEnum


logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"],
)

@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: str, push_request: PushRequest,
                        app_settings: Settings = Depends(get_settings)):
    project_model =await ProjectModel.create_instence(
        db_client=request.app.db_client,
    )
    chunk_model = await ChunkModel.create_instence(
        db_client=request.app.db_client,
    )
    project =await project_model.get_project_or_create_one(
        project_id=project_id
    )

    if not project:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"signal": ResponseSignal.PROJECT_NOT_FOUND.value})
    
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client
    )

    has_records=True
    page_no=1
    inserted_items_count=0
    idx=0

    while has_records:
        page_chunks = await chunk_model.get_project_chunks(project_id=project.id, page_no=page_no)
        if len(page_chunks):
            page_no+=1
        
        if not page_chunks or len(page_chunks)==0:
            has_records=False
            break
            
        chunks_ids = list(range(idx,idx+len(page_chunks)))
        idx += len(page_chunks)
    
        is_inserted= nlp_controller.index_into_vector_db(project=project, chunks=page_chunks,
                                             do_reset=push_request.do_reset,
                                             chunks_ids=chunks_ids
                                             )
        if not is_inserted:
            logger.error(f"Failed to index chunks into vector database for project_id: {project_id}, page_no: {page_no}")
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value})
        inserted_items_count+= len(page_chunks)
    return JSONResponse(status_code=status.HTTP_200_OK,
                         content={"signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
                                  "inserted_items_count": inserted_items_count})


@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info(request: Request, project_id: str):
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

    collection_info = nlp_controller.get_vector_db_collection_info(project=project)

    if collection_info is None:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"signal": ResponseSignal.COLLECTION_NOT_FOUND.value})
    
    return JSONResponse(status_code=status.HTTP_200_OK,
                         content={"signal": ResponseSignal.COLLECTION_INFO_RETRIEVE_SUCCESS.value,
                                  "collection_info": collection_info.dict()})
@nlp_router.post("/index/search/{project_id}")
async def search_index(request: Request, project_id: str, search_request: SearchRequest):
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

    search_results = nlp_controller.search_vector_db_collection(
        project=project,
        text=search_request.text,
        limit=search_request.limit
    )

    if not search_results:
        return JSONResponse(status_code=status.HTTP_200_OK, content={"signal": ResponseSignal.VECTOR_DB_SEARCH_ERROR.value, "results": []})
    
    return JSONResponse(status_code=status.HTTP_200_OK, content={"signal": ResponseSignal.VECTOR_DB_SEARCH_SUCCESS.value, "results": search_results})