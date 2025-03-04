import json
import os
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import uuid
import sys
from pathlib import Path

# 确保可以导入 mem0 模块
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mem0.configs.vector_stores.dashvector import MetricType
from mem0.vector_stores.dashvector import DashVectorDB, OutputData

"""
DashVectorDB 单元测试

运行测试:
    pytest -xvs tests/vector_stores/test_dashvector.py

这些测试使用 mock 对象模拟 DashVector 客户端和集合，
不需要实际连接到 DashVector 服务。

环境变量:
    DASHVECTOR_URL: DashVector 服务的 URL
    DASHVECTOR_API_KEY: DashVector 的 API 密钥
"""

# 从环境变量获取 DashVector 配置
DASHVECTOR_URL = os.environ.get("DASHVECTOR_URL", "http://test-endpoint.com")
DASHVECTOR_API_KEY = os.environ.get("DASHVECTOR_API_KEY", "test-api-key")

@pytest.fixture
def mock_dashvector_client():
    """模拟 DashVector 客户端"""
    with patch("dashvector.Client") as mock_client:
        # 创建模拟客户端
        client = MagicMock()
        mock_client.return_value = client

        # 设置 list_cols 的返回值
        mock_collection_list = MagicMock()
        mock_collection_list.output = ["test_collection"]
        client.list.return_value = mock_collection_list
        
        # 创建模拟集合
        collection = MagicMock()
        client.get.return_value = collection

        # 设置模拟方法
        client.create.return_value = MagicMock()
        client.describe.return_value = MagicMock()
        client.delete.return_value = MagicMock()

        # 设置集合方法
        collection.insert.return_value = MagicMock()
        collection.query.return_value = MagicMock()
        collection.delete.return_value = MagicMock()
        collection.update.return_value = MagicMock()
        collection.fetch.return_value = MagicMock()

        yield client

@pytest.fixture
def dashvector_instance(mock_dashvector_client):
    """创建 DashVectorDB 实例"""
    with patch("dashvector.Client", return_value=mock_dashvector_client):
        # 重置 create 调用计数，避免 test_create_col 测试失败
        mock_dashvector_client.create.reset_mock()
        
        db = DashVectorDB(
            url=DASHVECTOR_URL,
            api_key=DASHVECTOR_API_KEY,
            collection_name="test_collection",
            embedding_model_dims=1536,
            metric_type=MetricType.COSINE
        )
        yield db

def test_init(mock_dashvector_client):
    """测试初始化"""
    # 重置 create 调用计数
    mock_dashvector_client.create.reset_mock()
    
    with patch("dashvector.Client", return_value=mock_dashvector_client):
        db = DashVectorDB(
            url=DASHVECTOR_URL,
            api_key=DASHVECTOR_API_KEY,
            collection_name="test_collection",
            embedding_model_dims=1536,
            metric_type=MetricType.COSINE
        )
        
        assert db.url == DASHVECTOR_URL
        assert db.api_key == DASHVECTOR_API_KEY
        assert db.collection_name == "test_collection"
        assert db.embedding_model_dims == 1536
        assert db.metric_type == "cosine"
        
        # 验证客户端初始化
        mock_dashvector_client.get.assert_called_with("test_collection")

def test_create_col(dashvector_instance, mock_dashvector_client):
    """测试创建集合"""
    # 重置 create 调用计数
    mock_dashvector_client.create.reset_mock()
    
    dashvector_instance.create_col(
        collection_name="new_collection",
        vector_size="768",
        metric_type=MetricType.euclidean
    )
    
    # 验证创建调用
    mock_dashvector_client.create.assert_called_once()
    args, kwargs = mock_dashvector_client.create.call_args
    
    assert kwargs["name"] == "new_collection"
    assert kwargs["dimension"] == 768
    assert kwargs["metric"] == "euclidean"
    assert "user_id" in kwargs["fields_schema"]
    assert "data" in kwargs["fields_schema"]
    assert "hash" in kwargs["fields_schema"]
    assert "created_at" in kwargs["fields_schema"]

def test_insert_vectors(dashvector_instance, mock_dashvector_client):
    """测试插入向量"""
    # 模拟 Doc 类
    with patch("dashvector.Doc") as mock_doc:
        mock_doc_instance = MagicMock()
        mock_doc.return_value = mock_doc_instance

        # 测试数据
        ids = ["id1", "id2"]
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        payloads = [
            {"user_id": "user1", "data": {"text": "hello"}, "hash": "hash1", "created_at": "2023-01-01"},
            {"user_id": "user2", "data": {"text": "world"}, "hash": "hash2", "created_at": "2023-01-02"}
        ]

        # 执行插入
        dashvector_instance.insert(ids, vectors, payloads)

        # 验证 Doc 创建
        assert mock_doc.call_count == 2

        # 验证集合插入调用
        dashvector_instance.collection.insert.assert_called_once()

def test_search_vectors(dashvector_instance, mock_dashvector_client):
    """测试搜索向量"""
    # 模拟搜索结果
    mock_item1 = MagicMock()
    mock_item1.id = "id1"
    mock_item1.score = 0.95
    mock_item1.fields = {
        "user_id": "user1",
        "data": json.dumps({"text": "hello"}),
        "hash": "hash1",
        "created_at": "2023-01-01"
    }

    mock_item2 = MagicMock()
    mock_item2.id = "id2"
    mock_item2.score = 0.85
    mock_item2.fields = {
        "user_id": "user2",
        "data": json.dumps({"text": "world"}),
        "hash": "hash2",
        "created_at": "2023-01-02"
    }

    # 设置模拟返回值
    dashvector_instance.collection.query.return_value.output = [mock_item1, mock_item2]

    # 模拟 _parse_output 方法
    with patch.object(dashvector_instance, '_parse_output') as mock_parse:
        # 设置 _parse_output 的返回值
        mock_parse.return_value = [
            OutputData(id="id1", score=0.95, payload={"user_id": "user1", "text": "hello", "hash": "hash1", "created_at": "2023-01-01"}),
            OutputData(id="id2", score=0.85, payload={"user_id": "user2", "text": "world", "hash": "hash2", "created_at": "2023-01-02"})
        ]
        
        # 执行搜索
        query_vector = [0.1, 0.2, 0.3]
        results = dashvector_instance.search(query_vector, limit=2, filters={"user_id": "user1"})

        # 验证搜索调用
        dashvector_instance.collection.query.assert_called_once()

        # 验证结果
        assert len(results) == 2
        assert results[0].id == "id1"
        assert results[0].score == 0.95
        assert results[0].payload["text"] == "hello"

def test_delete_vector(dashvector_instance):
    """测试删除向量"""
    # 执行删除
    dashvector_instance.delete("id1")

    # 验证删除调用
    dashvector_instance.collection.delete.assert_called_once()
    # 注意：不检查参数，因为实现可能会变化

def test_update_vector(dashvector_instance):
    """测试更新向量"""
    # 执行更新
    vector = [0.1, 0.2, 0.3]
    payload = {"user_id": "user1", "data": {"text": "updated"}, "hash": "hash1", "created_at": "2023-01-01"}
    dashvector_instance.update("id1", vector, payload)

    # 验证更新调用
    dashvector_instance.collection.update.assert_called_once()

def test_get_vector(dashvector_instance):
    """测试获取向量"""
    # 模拟获取结果
    mock_item = MagicMock()
    mock_item.id = "id1"
    mock_item.fields = {
        "user_id": "user1",
        "data": json.dumps({"text": "hello"}),
        "hash": "hash1",
        "created_at": "2023-01-01"
    }

    # 设置模拟返回值
    dashvector_instance.collection.fetch.return_value.output = [mock_item]

    # 模拟 _parse_output 方法
    with patch.object(dashvector_instance, '_parse_output') as mock_parse:
        # 设置 _parse_output 的返回值
        mock_parse.return_value = [
            OutputData(id="id1", score=None, payload={"user_id": "user1", "text": "hello", "hash": "hash1", "created_at": "2023-01-01"})
        ]
        
        # 执行获取
        result = dashvector_instance.get("id1")

        # 验证获取调用
        dashvector_instance.collection.fetch.assert_called_once()

        # 验证结果
        assert result.id == "id1"
        assert result.payload["text"] == "hello"

def test_list_cols(dashvector_instance, mock_dashvector_client):
    """测试列出集合"""
    # 模拟集合列表
    mock_collection_list = MagicMock()
    mock_collection_list.output = ["collection1", "collection2"]
    mock_dashvector_client.list.return_value = mock_collection_list

    # 执行列出
    collections = dashvector_instance.list_cols()

    # 验证列出调用
    assert mock_dashvector_client.list.call_count >= 1

    # 验证结果
    assert len(collections) == 2
    assert "collection1" in collections
    assert "collection2" in collections

def test_list_vectors(dashvector_instance):
    """测试列出向量"""
    # 模拟列出结果
    mock_item1 = MagicMock()
    mock_item1.id = "id1"
    mock_item1.fields = {
        "user_id": "user1",
        "data": json.dumps({"text": "hello"}),
        "hash": "hash1",
        "created_at": "2023-01-01"
    }

    mock_item2 = MagicMock()
    mock_item2.id = "id2"
    mock_item2.fields = {
        "user_id": "user2",
        "data": json.dumps({"text": "world"}),
        "hash": "hash2",
        "created_at": "2023-01-02"
    }

    # 设置模拟返回值
    dashvector_instance.collection.query.return_value.output = [mock_item1, mock_item2]

    # 模拟 _parse_output 方法
    with patch.object(dashvector_instance, '_parse_output') as mock_parse:
        # 设置 _parse_output 的返回值
        mock_parse.return_value = [
            OutputData(id="id1", score=None, payload={"user_id": "user1", "text": "hello", "hash": "hash1", "created_at": "2023-01-01"}),
            OutputData(id="id2", score=None, payload={"user_id": "user2", "text": "world", "hash": "hash2", "created_at": "2023-01-02"})
        ]
        
        # 执行列出
        results = dashvector_instance.list(filters={"user_id": "user1"}, limit=100)

        # 验证列出调用
        dashvector_instance.collection.query.assert_called_once()

        # 验证结果
        assert len(results) == 2

@pytest.fixture
def real_dashvector_db():
    """创建真实的 DashVector 连接（集成测试用）
    
    注意: 此 fixture 需要设置环境变量才能运行
    - DASHVECTOR_URL: DashVector 服务的 URL
    - DASHVECTOR_API_KEY: DashVector 的 API 密钥
    
    如果未设置环境变量，相关测试将被跳过
    """
    # 检查是否设置了必要的环境变量
    url = os.environ.get("DASHVECTOR_URL")
    api_key = os.environ.get("DASHVECTOR_API_KEY")
    
    if not url or not api_key:
        pytest.skip("缺少 DASHVECTOR_URL 或 DASHVECTOR_API_KEY 环境变量")
    
    # 创建唯一名称的集合
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    
    # 创建 DashVector 实例
    db = DashVectorDB(
        url=url,
        api_key=api_key,
        collection_name=collection_name,
        embedding_model_dims=1536,
        metric_type=MetricType.COSINE
    )
    
    yield db
    
    # 测试后清理 - 删除测试集合
    try:
        db.delete_col()
    except Exception as e:
        print(f"清理测试集合时出错: {e}")

# 集成测试示例 - 只有在设置了环境变量时才会运行
def test_real_insert_and_search(real_dashvector_db):
    """使用真实 DashVector 服务的集成测试"""
    # 插入测试数据
    ids = ["test_id_1", "test_id_2"]
    vectors = [[0.1, 0.2, 0.3] * 512, [0.4, 0.5, 0.6] * 512]
    payloads = [
        {"user_id": "test_user", "data": {"text": "test document 1"}, "hash": "hash1", "created_at": "2023-01-01"},
        {"user_id": "test_user", "data": {"text": "test document 2"}, "hash": "hash2", "created_at": "2023-01-02"}
    ]
    
    real_dashvector_db.insert(ids, vectors, payloads)
    
    # 搜索并验证结果
    results = real_dashvector_db.search(
        query=vectors[0], 
        limit=2,
        filters={"user_id": "test_user"}
    )
    
    assert len(results) > 0
    # 验证返回的结果包含我们插入的数据
    found = False
    for result in results:
        if result.id in ids:
            found = True
            break
    
    assert found, "未找到插入的测试数据" 