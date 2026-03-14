from .BaseDataModel import BaseDataModel
from .db_schemes import project
from .enums.DataBaseEnum import DataBaseEnum

class ProjectModel(BaseDataModel):

    def __init__(self, db_client):
        super().__init__(db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_PROJECT_NAME.value]

    async def create_project(self, project: project):

        result = await self.collection.insert_one(project.dict(by_alias=True, exclude_unset=True))
        project.id = result.inserted_id
        return project
    

    async def get_project_or_create_one(self, project_id):

        record = await self.collection.find_one({"project_id": project_id})

        if record is None:
            new_project = project(project_id=project_id)
            new_project = await self.create_project(new_project)

            return new_project
        

        return project(**record)
    

    async def get_all_projects(self, page: int = 1, page_size: int = 10):

        # count total documents in the collection
        total_projects = await self.collection.count_documents({})

        # claculate total pages
        total_pages = total_projects // page_size
        if total_projects % page_size != 0:
            total_pages += 1

        cursor = self.collection.find().skip((page - 1) * page_size).limit(page_size)
        projects = []
        async for document in cursor:
            projects.append(project(**document))

        return projects, total_pages

        