from celery import chain
from celery_app import celery_app, get_setup_utils
from helpers.config import get_settings
import asyncio
from tasks.file_processing import process_project_files
from tasks.data_indexing import index_data_content

import logging
logger = logging.getLogger(__name__)

@celery_app.task(
                 bind=True, name="tasks.process_workflow.push_after_process_task",
                 autoretry_for=(Exception,),
                 retry_kwargs={'max_retries': 3, 'countdown': 60}
                )
def push_after_process_task(self, prev_task_result):
    """
    This task runs after file processing is complete.
    It triggers the indexing task with the project_id from the previous task.
    """
    project_id = prev_task_result.get("project_id")
    do_reset = prev_task_result.get("do_reset", 0)

    logger.info(f"[ProcessWorkflow] Starting indexing for project {project_id} (do_reset={do_reset})")

    # Call the indexing task directly (it will handle its own async execution)
    indexing_task = index_data_content.delay(project_id=project_id, do_reset=do_reset)
    
    return {
        "project_id": project_id,
        "do_reset": do_reset,
        "indexing_task_id": indexing_task.id,
        "signal": "INDEXING_STARTED"
    }


@celery_app.task(
                 bind=True, name="tasks.process_workflow.process_and_push_workflow",
                 autoretry_for=(Exception,),
                 retry_kwargs={'max_retries': 3, 'countdown': 60}
                )
def process_and_push_workflow(  self, project_id: int, 
                                file_id: int, chunk_size: int,
                                overlap_size: int, do_reset: int):
    """
    Workflow that chains file processing and indexing tasks.
    
    Flow:
    1. Process files (extract, chunk, store in DB)
    2. Index chunks into vector database
    """
    logger.info(f"[ProcessWorkflow] Starting workflow for project {project_id}")

    workflow = chain(
        process_project_files.s(project_id, file_id, chunk_size, overlap_size, do_reset),
        push_after_process_task.s()
    )

    result = workflow.apply_async()

    return {
        "signal": "WORKFLOW_STARTED",
        "workflow_id": result.id,
        "project_id": project_id,
        "tasks": [
            "tasks.file_processing.process_project_files", 
            "tasks.data_indexing.index_data_content"
        ]
    }