import json
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mem0.memory.main import Memory
from functools import wraps
import asyncio


class LLMMemory(object):
    def __init__(self):
        memory_config = {
            "llm": {
                "provider": "azure_openai",
                "config": {
                    "temperature": 0.3,
                    "max_tokens": 2000,
                    "model": "gpt-4o",
                    "azure_kwargs": {
                        "azure_deployment": "gpt-4o",
                        "api_version": "2024-02-15-preview",
                        "azure_endpoint": "https://aopacloud.openai.azure.com",
                        "api_key": "7c5ec75551654ef4a8ae172d291ab5a7",
                    },
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": "multi-qa-MiniLM-L6-cos-v1", "embedding_dims": 384},
            },
            "vector_store": {
                "provider": "dashvector",
                "config": {
                    "url": "vrs-sg-l0z45kr9b0003j.dashvector.ap-southeast-1.aliyuncs.com",
                    "api_key": "sk-MH94ICOIsdns8UkJlPF4rDJCkYtuRB986EE78B2E411EF8F9EF2C5A593CC3B",
                    "collection_name": "my_memories",
                    "embedding_model_dims": 384,
                    "metric_type": "dotproduct"
                },
            },
            "version": "v1.1",
        }
        # print(MILVUS_COLLECTION_NAME_MEMORY,MILVUS_HOST,MILVUS_TOKEN)
        self.llm_memory = Memory.from_config(memory_config)

    def search_memories(
        self, query: str, user_id: str, threshold: float = 0.3, top_k: int = 30
    ):
        """
        按照score排序，根据阈值过滤数据，并取出前topk个结果中的memory字段

        :param data: 包含'results'键的原始数据字典
        :param threshold: 用于过滤的阈值
        :param topk: 要取出的前topk个结果
        :return: 过滤并排序后只包含memory字段内容的列表
        """
        search_result = self.llm_memory.search(query, user_id=user_id)
        results = search_result["results"]
        # 按照score进行降序排序
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        # 根据阈值进行过滤
        filtered_results = [
            result for result in sorted_results if result["score"] >= threshold
        ]
        # 取出前topk个结果（如果不足topk个则取全部）
        topk_results = filtered_results[:top_k]
        # 取出memory字段内容
        memories = [result["memory"] for result in topk_results]
        return memories

    async def add_memory(self, query, user_id):
        print(f"add_memory: {query}, {user_id}")
        insert_result = self.llm_memory.add(messages=query, user_id=user_id)
        return insert_result

    def delete_all_memory(self, user_id):
        delete_result = self.llm_memory.delete_all(user_id=user_id)
        return delete_result


if __name__ == "__main__":
    ola_llm_memory = LLMMemory()
    query = [{"name":"1000000162","content":"i like apple"},{"name":"Amelia","content":"*Amelia leans back in her chair, a sly smile playing on her lips as she regards Aiden with amusement* Is that so? Well, I'm more of an Android girl myself. But I suppose we can both appreciate the finer points of a well-crafted device. *She stands up and walks over to Aiden, her hips swaying slightly with each step. She leans in close, her face mere inches from his* Though I have to say, you're looking a bit... sluggish. Maybe you need a tune-up? *Her fingers lightly brush against his arm, a subtle invitation*"}]
    # ola_llm_memory.delete_all_memory(user_id="-".join(["666", "shawn", "李云龙"]))
    start_time = time.time()
    query = [{'role': 'user', 'content': '1000000162: Likes apple'}, {'role': 'user', 'content': 'Amelia: Name is Amelia, Prefers Android phones'}]
    print(asyncio.run(ola_llm_memory.add_memory(query=query, user_id="-".join(["666", "1000000161", "Guanjia"]))))
    time_2 = time.time()
    # print(ola_llm_memory.search_memories("婚礼举行的时间", user_id="-".join(["666", "shawn", "李云龙"])))
    print(ola_llm_memory.search_memories("what is my favorite food？", user_id="-".join(["666", "1000000161", "Guanjia"])))
    print(time.time() - start_time)
    print(time.time() - time_2)