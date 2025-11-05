# scripts/evaluate_model.py
import os
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)

class ModelEvaluator:
    def __init__(self, model_path: str, tokenizer_path: str = None):
        """初始化评测器"""
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        
        print(f"加载模型: {model_path}")
        self.load_model()
        
        # 初始化指标计算器（避免依赖问题）
        self.metrics_initialized = False
        self.init_metrics()
    
    def init_metrics(self):
        """初始化评估指标，处理依赖问题"""
        try:
            import evaluate
            self.bleu_metric = evaluate.load("bleu")
            self.rouge_metric = evaluate.load("rouge")
            self.metrics_initialized = True
            print("✅ 评估指标加载成功")
        except ImportError as e:
            print(f"⚠️  评估指标依赖缺失: {e}")
            print("🔧 请安装依赖: pip install evaluate rouge-score nltk absl-py")
            self.metrics_initialized = False
        except Exception as e:
            print(f"⚠️  评估指标初始化失败: {e}")
            self.metrics_initialized = False
    
    def load_model(self):
        """加载模型和tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, 
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            print("✅ 模型加载成功")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """生成回复"""
        try:
            # 直接使用prompt，不应用chat模板
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 解码回复
            response = self.tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"生成回复时出错: {e}")
            return ""
    
    def extract_reasoning_and_answer(self, response: str) -> tuple:
        """从回复中提取推理过程和最终答案"""
        reasoning = ""
        answer = ""
        
        if "<reasoning>" in response and "</reasoning>" in response:
            # 提取推理部分
            start_idx = response.find("<reasoning>") + len("<reasoning>")
            end_idx = response.find("</reasoning>")
            reasoning = response[start_idx:end_idx].strip()
            
            # 提取答案部分
            answer_start = response.find("答：")
            if answer_start != -1:
                answer = response[answer_start + len("答："):].strip()
        else:
            # 如果没有特定格式，尝试分割
            if "答：" in response:
                answer_start = response.find("答：")
                reasoning = response[:answer_start].strip()
                answer = response[answer_start + len("答："):].strip()
            else:
                answer = response
        
        return reasoning, answer
    
    def calculate_similarity_metrics(self, reference: str, prediction: str) -> Dict:
        """计算相似度指标（简化版，避免依赖问题）"""
        metrics = {}
        
        if not self.metrics_initialized:
            # 使用简单的字符串匹配作为备选
            metrics.update(self.calculate_basic_metrics(reference, prediction))
            return metrics
        
        try:
            # BLEU score
            bleu_result = self.bleu_metric.compute(
                predictions=[prediction],
                references=[[reference]]
            )
            metrics["bleu"] = float(bleu_result["bleu"])  # 转换为Python float
            
            # ROUGE score
            rouge_result = self.rouge_metric.compute(
                predictions=[prediction],
                references=[reference]
            )
            metrics["rouge1"] = float(rouge_result["rouge1"])  # 转换为Python float
            metrics["rouge2"] = float(rouge_result["rouge2"])  # 转换为Python float
            metrics["rougeL"] = float(rouge_result["rougeL"])  # 转换为Python float
            
        except Exception as e:
            print(f"计算相似度指标时出错，使用基础指标: {e}")
            metrics.update(self.calculate_basic_metrics(reference, prediction))
        
        return metrics
    
    def calculate_basic_metrics(self, reference: str, prediction: str) -> Dict:
        """计算基础相似度指标（不依赖外部包）"""
        ref_words = set(reference.split())
        pred_words = set(prediction.split())
        
        # 计算Jaccard相似度
        intersection = len(ref_words.intersection(pred_words))
        union = len(ref_words.union(pred_words))
        jaccard_similarity = float(intersection / union if union > 0 else 0)
        
        # 计算重叠率
        overlap_ratio = float(len([w for w in prediction.split() if w in reference]) / len(prediction.split()) if prediction else 0)
        
        return {
            "jaccard_similarity": jaccard_similarity,
            "overlap_ratio": overlap_ratio,
            "bleu": jaccard_similarity,  # 用Jaccard近似BLEU
            "rouge1": overlap_ratio,     # 用重叠率近似ROUGE-1
            "rouge2": 0.0,               # 简化处理
            "rougeL": overlap_ratio      # 用重叠率近似ROUGE-L
        }
    
    def evaluate_single_example(self, test_case: Dict) -> Dict:
        """评估单个测试用例"""
        question = test_case["input"].replace("用户：", "").strip()
        expected_output = test_case["output"]
        
        # 生成回复
        start_time = datetime.now()
        generated_response = self.generate_response(question)
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # 提取推理和答案
        reasoning, answer = self.extract_reasoning_and_answer(generated_response)
        
        # 计算指标
        similarity_metrics = self.calculate_similarity_metrics(expected_output, generated_response)
        
        # 计算响应长度
        response_length = len(generated_response)
        
        return {
            "question": question,
            "expected_output": expected_output,
            "generated_response": generated_response,
            "reasoning": reasoning,
            "answer": answer,
            "generation_time": float(generation_time),  # 转换为Python float
            "response_length": int(response_length),    # 转换为Python int
            "similarity_metrics": similarity_metrics,
            "has_reasoning_format": "<reasoning>" in generated_response and "</reasoning>" in generated_response,
            "has_answer_format": "答：" in generated_response,
            "is_empty_response": len(generated_response.strip()) == 0
        }
    
    def evaluate_on_dataset(self, test_file: str, num_samples: int = None) -> Dict:
        """在测试集上进行评估"""
        print(f"开始评估，测试文件: {test_file}")
        
        # 检查测试文件是否存在
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"测试文件不存在: {test_file}")
        
        # 加载测试数据
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line.strip()))
        
        if num_samples and num_samples < len(test_data):
            test_data = test_data[:num_samples]
        
        print(f"测试样本数量: {len(test_data)}")
        
        results = []
        total_metrics = {
            "generation_time": [],
            "response_length": [],
            "has_reasoning_format": 0,
            "has_answer_format": 0,
            "empty_responses": 0
        }
        
        # 初始化指标累计
        metric_keys = ["bleu", "rouge1", "rouge2", "rougeL", "jaccard_similarity", "overlap_ratio"]
        for key in metric_keys:
            total_metrics[key] = []
        
        for i, test_case in enumerate(test_data):
            print(f"处理样本 {i+1}/{len(test_data)}")
            
            try:
                result = self.evaluate_single_example(test_case)
                results.append(result)
                
                # 累计指标
                metrics = result["similarity_metrics"]
                for key in metrics:
                    if key in total_metrics:
                        total_metrics[key].append(metrics[key])
                
                # 累计其他指标
                total_metrics["generation_time"].append(result["generation_time"])
                total_metrics["response_length"].append(result["response_length"])
                total_metrics["has_reasoning_format"] += int(result["has_reasoning_format"])
                total_metrics["has_answer_format"] += int(result["has_answer_format"])
                total_metrics["empty_responses"] += int(result["is_empty_response"])
                        
            except Exception as e:
                print(f"评估样本 {i+1} 时出错: {e}")
                continue
        
        # 计算平均指标 - 确保所有值都是Python原生类型
        avg_metrics = {}
        for key, values in total_metrics.items():
            if isinstance(values, list) and values:
                # 转换为Python原生类型
                avg_metrics[f"avg_{key}"] = float(np.mean(values))
                avg_metrics[f"std_{key}"] = float(np.std(values))
                avg_metrics[f"min_{key}"] = float(np.min(values))
                avg_metrics[f"max_{key}"] = float(np.max(values))
            else:
                # 对于非列表值（计数类型），直接使用
                avg_metrics[key] = values
        
        # 计算格式正确率
        if results:
            avg_metrics["reasoning_format_rate"] = float(total_metrics["has_reasoning_format"] / len(results))
            avg_metrics["answer_format_rate"] = float(total_metrics["has_answer_format"] / len(results))
            avg_metrics["empty_response_rate"] = float(total_metrics["empty_responses"] / len(results))
            avg_metrics["total_samples"] = int(len(results))  # 转换为Python int
        else:
            avg_metrics.update({
                "reasoning_format_rate": 0.0,
                "answer_format_rate": 0.0,
                "empty_response_rate": 0.0,
                "total_samples": 0
            })
        
        return {
            "results": results,
            "summary": avg_metrics,
            "total_samples": len(results),
            "metrics_available": self.metrics_initialized
        }
    
    def convert_to_serializable(self, obj):
        """将对象转换为JSON可序列化的格式"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def run_comprehensive_evaluation(self, test_file: str, output_dir: str, num_samples: int = None):
        """运行综合评估"""
        print("开始综合评估...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 评估结果
        evaluation_result = self.evaluate_on_dataset(test_file, num_samples)
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 转换为可序列化的格式
        serializable_result = self.convert_to_serializable(evaluation_result)
        
        # 保存详细结果
        detailed_file = os.path.join(output_dir, f"detailed_results_{timestamp}.json")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        # 保存摘要结果
        summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result["summary"], f, indent=2, ensure_ascii=False)
        
        # 生成CSV报告
        self.generate_csv_report(serializable_result["results"], output_dir, timestamp)
        
        # 打印评估摘要
        self.print_evaluation_summary(serializable_result["summary"], serializable_result["metrics_available"])
        
        print(f"✅ 评估完成!")
        print(f"📊 详细结果: {detailed_file}")
        print(f"📈 评估摘要: {summary_file}")
        
        return evaluation_result
    
    def generate_csv_report(self, results: List[Dict], output_dir: str, timestamp: str):
        """生成CSV格式的报告"""
        csv_data = []
        
        for i, result in enumerate(results):
            row = {
                "id": i + 1,
                "question": result["question"],
                "expected_output": result["expected_output"],
                "generated_response": result["generated_response"],
                "reasoning": result["reasoning"],
                "answer": result["answer"],
                "generation_time": result["generation_time"],
                "response_length": result["response_length"],
                "has_reasoning_format": result["has_reasoning_format"],
                "has_answer_format": result["has_answer_format"],
                "is_empty_response": result["is_empty_response"]
            }
            
            # 添加相似度指标
            for metric_name, value in result["similarity_metrics"].items():
                row[metric_name] = value
            
            csv_data.append(row)
        
        csv_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"📋 CSV报告: {csv_file}")
    
    def print_evaluation_summary(self, summary: Dict, metrics_available: bool):
        """打印评估摘要"""
        print("\n" + "="*60)
        print("📊 模型评估摘要")
        print("="*60)
        
        print(f"🤖 模型路径: {self.model_path}")
        print(f"📈 评估样本数: {summary.get('total_samples', 'N/A')}")
        print(f"📊 指标状态: {'完整指标' if metrics_available else '基础指标'}")
        print()
        
        if metrics_available:
            print("🎯 相似度指标:")
            print(f"   BLEU Score: {summary.get('avg_bleu', 0):.4f} ± {summary.get('std_bleu', 0):.4f}")
            print(f"   ROUGE-1:    {summary.get('avg_rouge1', 0):.4f} ± {summary.get('std_rouge1', 0):.4f}")
            print(f"   ROUGE-2:    {summary.get('avg_rouge2', 0):.4f} ± {summary.get('std_rouge2', 0):.4f}")
            print(f"   ROUGE-L:    {summary.get('avg_rougeL', 0):.4f} ± {summary.get('std_rougeL', 0):.4f}")
        else:
            print("🎯 基础相似度指标:")
            print(f"   Jaccard相似度: {summary.get('avg_jaccard_similarity', 0):.4f}")
            print(f"   重叠率:        {summary.get('avg_overlap_ratio', 0):.4f}")
        
        print()
        print("⏱️  性能指标:")
        print(f"   平均生成时间: {summary.get('avg_generation_time', 0):.2f}秒")
        print(f"   平均响应长度: {summary.get('avg_response_length', 0):.1f}字符")
        print()
        
        print("📝 格式正确率:")
        print(f"   推理格式正确率: {summary.get('reasoning_format_rate', 0)*100:.1f}%")
        print(f"   答案格式正确率: {summary.get('answer_format_rate', 0)*100:.1f}%")
        print(f"   空响应率:       {summary.get('empty_response_rate', 0)*100:.1f}%")
        print("="*60)

def main():
    """主函数"""
    # 配置参数 - 更新路径
    MODEL_PATH = "./models/deepseek_r1_1.5b_lora/best_model"  # 要评估的模型路径
    TEST_FILE = "./dataset/sft_r1_val.jsonl"  # 测试数据文件
    OUTPUT_DIR = "./scripts/compare/evaluation_results"  # 输出目录
    NUM_SAMPLES = 10  # 评估样本数量 (None表示全部)
    
    # 检查测试文件是否存在
    if not os.path.exists(TEST_FILE):
        print(f"❌ 测试文件不存在: {TEST_FILE}")
        print("请检查文件路径是否正确")
        return
    
    # 检查输出目录是否存在，不存在则创建
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 单个模型评估
    print("开始单个模型评估...")
    try:
        evaluator = ModelEvaluator(MODEL_PATH)
        evaluation_result = evaluator.run_comprehensive_evaluation(TEST_FILE, OUTPUT_DIR, NUM_SAMPLES)
        
        # 显示几个示例
        print("\n🔍 示例输出:")
        for i, result in enumerate(evaluation_result["results"][:3]):
            print(f"\n示例 {i+1}:")
            print(f"问题: {result['question']}")
            print(f"生成回复: {result['generated_response']}")
            print(f"推理部分: {result['reasoning']}")
            print(f"答案部分: {result['answer']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()