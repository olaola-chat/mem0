import json
import logging
from typing import Dict, List, Optional, Any

from pydantic import BaseModel

from mem0.configs.vector_stores.dashvector import MetricType
from mem0.vector_stores.base import VectorStoreBase

try:
    import dashvector  # noqa: F401
except ImportError:
    raise ImportError("The 'dashvector' library is required. Please install it using 'pip install dashvector'.")

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: Optional[str]  # memory id
    score: Optional[float]  # distance
    payload: Optional[Dict]  # metadata


class DashVectorDB(VectorStoreBase):
    """DashVector implementation for Mem0."""

    def __init__(
        self,
        url: str,
        api_key: str,
        collection_name: str,
        embedding_model_dims: int,
        metric_type: MetricType,
    ) -> None:
        """
        Initialize the DashVectorDB database.
        
        Args:
            url (str): Full URL for DashVector server.
            api_key (str): API key for DashVector server.
            collection_name (str): Name of the collection (defaults to mem0).
            embedding_model_dims (int): Dimensions of the embedding model (defaults to 1536).
            metric_type (MetricType): Metric type for similarity search (defaults to L2).
        """
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        
        # 转换 MetricType 到 DashVector 支持的格式
        dash_metric_map = {
            MetricType.euclidean: "euclidean",
            MetricType.dotproduct: "dotproduct",
            MetricType.COSINE: "cosine"
        }
        self.metric_type = dash_metric_map.get(metric_type, "cosine")
        
        # 初始化 DashVector 客户端
        self.client = dashvector.Client(api_key=api_key, endpoint=url)
        
        # 检查是否已存在该集合，若不存在则创建
        collections = self.list_cols()
        if collection_name not in collections:
            self.create_col(collection_name, embedding_model_dims, metric_type)
        
        # 获取当前集合
        self.collection = self.client.get(collection_name)

    def create_col(
        self,
        collection_name: str,
        vector_size: str,
        metric_type: MetricType = MetricType.COSINE,
    ) -> None:
        """创建新的Collection
        
        Args:
            collection_name (str): Collection的名称
            vector_size (str): 向量维度
            metric_type (MetricType): 相似度计算方式 
        """
        # 使用小写值
        metric_type_value = metric_type.value.lower()
        
        # 定义字段 schema，包括必要的字段
        fields_schema = {
            "user_id": str,
            "data": str,
            "hash": str,
            "created_at": str
        }
        
        try:
            create_result = self.client.create(
                name=collection_name,
                dimension=int(vector_size),
                metric=metric_type_value,
                fields_schema=fields_schema
            )
            logging.info(f"Collection {collection_name} 创建成功")
        except Exception as e:
            logging.error(f"创建Collection失败: {e}")
            raise

    def insert(self, ids, vectors, payloads, **kwargs: Optional[dict[str, any]]):
        """向集合中插入向量
        
        Args:
            ids: 向量ID列表
            vectors: 向量数据列表
            payloads: 元数据列表
        """
        from dashvector import Doc
        
        if not ids or not vectors:
            logger.error("插入失败：ID或向量数据为空")
            return
        
        docs = []
        for i, (vector_id, vector, payload) in enumerate(zip(ids, vectors, payloads)):
            # 从 payload 中提取各个字段，如果不存在则使用默认值
            fields = {
                "user_id": payload.get("user_id", ""),
                "data": payload.get("data", ""),  # 直接存储 data 字段，不进行 JSON 序列化
                "hash": payload.get("hash", ""),
                "created_at": payload.get("created_at", "")
            }
            
            # 创建 Doc 对象
            doc = Doc(id=vector_id, vector=vector, fields=fields)
            docs.append(doc)
        
        try:
            insert_result = self.collection.insert(docs)
            logger.info(f"成功插入 {len(docs)} 条向量到集合 {self.collection_name}")
            return insert_result
        except Exception as e:
            logger.error(f"插入向量时出错: {e}")
            raise

    def _create_filter(self, filters: dict):
        """创建过滤条件字符串
        
        Args:
            filters (Dict): 过滤条件
            
        Returns:
            str: SQL WHERE 子句格式的过滤条件字符串
        """
        if not filters:
            return None
            
        filter_parts = []
        
        # 处理 user_id 过滤
        if "user_id" in filters:
            filter_parts.append(f"user_id = '{filters['user_id']}'")
        
        # 处理 metadata 过滤 (向后兼容)
        if "metadata" in filters:
            for key, value in filters["metadata"].items():
                if key == "user_id" and "user_id" not in filters:
                    # 如果 metadata 中有 user_id 并且顶级中没有，则添加过滤
                    if isinstance(value, str):
                        filter_parts.append(f"user_id = '{value}'")
                # 其他 metadata 字段可能需要搜索 data 中的内容
                elif isinstance(value, str):
                    filter_parts.append(f"data LIKE '%\"{key}\":\"{value}\"%'")
                elif isinstance(value, (int, float)):
                    filter_parts.append(f"data LIKE '%\"{key}\":{value}%'")
                elif isinstance(value, bool):
                    bool_value = "true" if value else "false"
                    filter_parts.append(f"data LIKE '%\"{key}\":{bool_value}%'")
        
        # 将所有条件用 AND 连接
        if filter_parts:
            return " AND ".join(filter_parts)
        
        return None

    def _parse_output(self, data):
        """Parse search results into OutputData objects.
        
        Args:
            data: Search results from DashVector.
            
        Returns:
            List[OutputData]: Parsed results.
        """
        results = []
        if not isinstance(data, list):
            return results
            
        for item in data:
            try:
                # 提取基本信息
                item_id = getattr(item, "id", None)
                score = getattr(item, "score", None)
                fields = getattr(item, "fields", {})
                # 直接使用字段作为 payload
                payload = dict(fields)
                
                # 创建 OutputData 对象
                output = OutputData(id=item_id, score=score, payload=payload)
                results.append(output)
                
            except Exception as e:
                logger.error(f"解析结果时出错: {e}")
                continue
        
        return results

    def search(self, query: list, limit: int = 5, filters: dict = None) -> list:
        """
        Search for similar vectors.
        
        Args:
            query (List[float]): Query vector.
            limit (int, optional): Number of results to return. Defaults to 5.
            filters (Dict, optional): Filters to apply to the search. Defaults to None.
            
        Returns:
            list: Search results.
        """
        filter_expr = self._create_filter(filters)
        # 执行查询
        results = self.collection.query(
            vector=query,
            topk=limit,
            filter=filter_expr
        )
        return self._parse_output(results.output)

    def delete(self, vector_id):
        """
        Delete a vector by ID.
        
        Args:
            vector_id (str): ID of the vector to delete.
        """
        self.collection.delete(ids=[vector_id])
        logger.info(f"Deleted vector {vector_id} from collection {self.collection_name}.")

    def update(self, vector_id=None, vector=None, payload=None):
        """
        Update a vector and its payload.
        
        Args:
            vector_id (str): ID of the vector to update.
            vector (List[float], optional): Updated vector.
            payload (Dict, optional): Updated payload.
        """
        update_data = {"id": vector_id}
        
        if vector:
            update_data["vector"] = vector
        
        if payload:
            fields = {
                "user_id": payload.get("user_id", ""),
                "data": payload.get("data", ""),
                "hash": payload.get("hash", ""),
                "created_at": payload.get("created_at", ""),
                "updated_at": payload.get("updated_at", "")
            }
            update_data["fields"] = fields
        
        self.collection.update([update_data])
        logger.info(f"Updated vector {vector_id} in collection {self.collection_name}.")

    def get(self, vector_id):
        """
        Retrieve a vector by ID.
        
        Args:
            vector_id (str): ID of the vector to retrieve.
            
        Returns:
            OutputData: Retrieved vector.
        """
        try:
            result = self.collection.fetch(ids=[vector_id])
            if not result or not result.output:
                logger.warning(f"Vector with ID {vector_id} not found")
                return None

            # 解析结果
            parsed_data = self._parse_output([result.output])
            if not parsed_data:
                logger.warning(f"Failed to parse data for vector ID {vector_id}")
                return None

            return parsed_data[0]
        except Exception as e:
            logger.error(f"Error retrieving vector {vector_id}: {e}")
            return None

    def list_cols(self):
        """
        List all collections.
        
        Returns:
            List[str]: List of collection names.
        """
        response = self.client.list()
        return response.output

    def delete_col(self):
        """
        Delete a collection.
        """
        self.client.delete(name=self.collection_name)
        logger.info(f"Collection {self.collection_name} deleted.")

    def col_info(self):
        """
        Get information about a collection.
        
        Returns:
            Dict[str, Any]: Collection information.
        """
        response = self.client.describe(name=self.collection_name)
        return response

    def list(self, filters: dict = None, limit: int = 100) -> list:
        """
        List all vectors in a collection.
        
        Args:
            filters (Dict, optional): Filters to apply to the list.
            limit (int, optional): Number of vectors to return. Defaults to 100.
            
        Returns:
            List[List[OutputData]]: List of vectors wrapped in a list.
        """
        filter_expr = self._create_filter(filters)
        
        # 在 DashVector 中，可以使用 query 不带 query_vector 参数来列出所有向量
        results = self.collection.query(
            filter=filter_expr,
            topk=limit,
            include_vector=True
        )
        # 将结果包装在列表中返回，以匹配其他 vector store 的实现
        return [self._parse_output(results.output)]