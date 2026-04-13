from fastapi import FastAPI, APIRouter, Depends, UploadFile, status , Request
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController, NLPController
import aiofiles
from models import ResponseSignal
import logging

from .schemes.data import ProcessRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel
from models.db_schemes import DataChunk
from models.db_schemes import Asset
from models.enums.AssetTypeEnum import AssetTypeEnum
from tasks.file_processing import process_project_files
from tasks.process_workflow import process_and_push_workflow


logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1", "data"],
)

@data_router.post("/upload/{project_id}")
async def upload_data(request: Request, project_id: int, file: UploadFile,
                      app_settings: Settings = Depends(get_settings)):
        
    project_model = await ProjectModel.create_instence(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)
    # validate the file properties
    data_controller = DataController()

    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": result_signal
            }
        )

    project_dir_path = ProjectController().get_project_path(project_id=project_id)
    file_path, file_id = data_controller.generate_unique_filepath(
        orig_file_name=file.filename,
        project_id=project_id
    )

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:

        logger.error(f"Error while uploading file: {e}")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.FILE_UPLOAD_FAILED.value
            }
        )
    
    # store file metadata in the database
    asset_model = await AssetModel.create_instence(db_client=request.app.db_client)
    asset_resource = Asset(
        asset_project_id=project.project_id,
        asset_type=AssetTypeEnum.FILE.value,
        asset_name = file_id,
        asset_size=os.path.getsize(file_path)
    )
    asset_record = await asset_model.create_asset(asset_resource)

    return JSONResponse(
            content={
                "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
                "file_id": str(asset_record.asset_name),
            }
        )

@data_router.post("/process/{project_id}")
async def process_request(request: Request, project_id:int , process_request: ProcessRequest):


    chunk_size = process_request.chunck_size
    overlap_size = process_request.overlap_size
    do_reset = process_request.do_reset

    # task = process_project_files.delay(
    #     project_id=project_id,
    #     file_id=process_request.file_id,
    #     chunk_size=chunk_size,
    #     overlap_size=overlap_size,
    #     do_reset=do_reset
    # )

    # return JSONResponse(
    #     content={
    #         "signal": ResponseSignal.PROCESSING_SUCCESS.value,
    #         "task_id": task.id,
    #     }
    # )

    project_model = await ProjectModel.create_instence(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    asset_model = await AssetModel.create_instence(
            db_client=request.app.db_client
        )
    
    nlp_controller = NLPController(
        embedding_client=request.app.embedding_client,
        generation_client=request.app.generation_client,
        vectordb_client=request.app.vectordb_client,
        template_parser=request.app.template_parser
    )

    project_files_ids = {}
    if process_request.file_id:
        asset_record = await asset_model.get_asset_record(
            asset_project_id=project.project_id,
            asset_id=process_request.file_id
        )

        if asset_record is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.FILE_ID_ERROR.value,
                }
            )

        project_files_ids = {
            asset_record.asset_id: asset_record.asset_name
        }
    
    else:
        
        project_files = await asset_model.get_all_project_assets(
            asset_project_id=project.project_id,
            asset_type=AssetTypeEnum.FILE.value,
        )

        project_files_ids = {
            record.asset_id: record.asset_name
            for record in project_files
        }

    if len(project_files_ids) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.NO_FILES_ERROR.value,
            }
        )

    process_controller = ProcessController(project_id=project_id)

    inserted_count = 0
    no_files_count = 0

    chunk_model = await ChunkModel.create_instence(db_client=request.app.db_client)


    if do_reset==1:
                
                # delete associated vector db collection
                collection_name = nlp_controller.create_collection_name(project_id=project.project_id)
                _ = await request.app.vectordb_client.delete_collection(collection_name=collection_name)
                
                # delete associated chunks in the database
                deleted_count = await chunk_model.delete_chunks_by_project_id(project_id=project.project_id)
        
                logger.info(f"Deleted {deleted_count} chunks for project_id: {project_id}")
        
    for asset_id,file_id in project_files_ids.items():
        file_content = process_controller.get_file_content(file_id=file_id)

        if file_content is None:
            logger.warning(f"File with id {file_id} not found for project_id: {project_id}")
            continue
        
        file_chunks = process_controller.process_file_content(file_id=file_id, file_content=file_content, chunk_size=chunk_size, overlap_size=overlap_size)

        if file_chunks is None or len(file_chunks) == 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.PROCESSING_FAILED.value
                }
            )
        
        file_chunks_records = [
            DataChunk(
                chunk_text=chunk.page_content,
                chunk_metadata=chunk.metadata,
                chunk_order= i+1,
                chunk_project_id= project.project_id,
                chunk_asset_id= asset_id

            )
            for i, chunk in enumerate(file_chunks)
            
            ]
        

        
        # chunk_model = await ChunkModel.create_instence(db_client=request.app.db_client)
        inserted_count += await chunk_model.insert_many_chunks(chunks=file_chunks_records)
        no_files_count += 1
    return JSONResponse(
        content={
            "signal": ResponseSignal.PROCESSING_SUCCESS.value,
            "inserted_count": inserted_count,
            "processed_files": no_files_count
        }
    )


@data_router.post("/process-and-push/{project_id}")
async def process_and_push_request(request: Request, project_id:int , process_request: ProcessRequest):


    chunk_size = process_request.chunck_size
    overlap_size = process_request.overlap_size
    do_reset = process_request.do_reset

    workflow_task = process_and_push_workflow.delay(
        project_id=project_id,
        file_id=process_request.file_id,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        do_reset=do_reset
    )

    return JSONResponse(
        content={
            "signal": ResponseSignal.PROCESS_AND_PUSH_WORKFLOW_READY.value,
            "task_id": workflow_task.id,
        }
    )

