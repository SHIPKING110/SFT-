# scripts/chat_with_model.py
import os
import torch
import json
from datetime import datetime
from typing import List, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

class ChatBot:
    def __init__(self, model_path: str):
        """初始化聊天机器人"""
        self.model_path = model_path
        self.conversation_history = []
        
        print(f"🤖 加载模型: {model_path}")
        self.load_model()
        print("✅ 模型加载完成，可以开始对话了！")
        print("💡 输入 '退出' 或 'quit' 结束对话")
        print("💡 输入 '清除' 或 'clear' 清除对话历史")
        print("💡 输入 '保存' 或 'save' 保存对话记录")
        print("-" * 60)
    
    def load_model(self):
        """加载模型和tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
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
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def format_conversation(self, messages: List[Dict]) -> str:
        """格式化对话历史"""
        formatted_text = ""
        for msg in messages:
            if msg["role"] == "user":
                formatted_text += f"用户：{msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted_text += f"助手：{msg['content']}\n"
        return formatted_text.strip()
    
    def generate_response(self, user_input: str, max_length: int = 1024) -> str:
        """生成回复"""
        try:
            # 添加用户输入到对话历史
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # 格式化对话
            prompt = self.format_conversation(self.conversation_history) + "\n助手："
            
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
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
            response = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            response = response.strip()
            
            # 添加到对话历史
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            print(f"生成回复时出错: {e}")
            return "抱歉，我遇到了一些问题，请重新尝试。"
    
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
    
    def save_conversation(self, filename: str = None):
        """保存对话记录"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
        
        # 确保目录存在
        os.makedirs("./conversations", exist_ok=True)
        filepath = os.path.join("./conversations", filename)
        
        conversation_data = {
            "model_path": self.model_path,
            "timestamp": datetime.now().isoformat(),
            "conversation": self.conversation_history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 对话已保存到: {filepath}")
        return filepath
    
    def load_conversation(self, filepath: str):
        """加载对话记录"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            self.conversation_history = conversation_data["conversation"]
            print(f"📂 已加载对话记录，共 {len(self.conversation_history)} 条消息")
            
            # 显示最近的几条消息
            print("\n最近的对话:")
            for msg in self.conversation_history[-4:]:  # 显示最后4条
                role = "用户" if msg["role"] == "user" else "助手"
                print(f"  {role}: {msg['content'][:100]}...")
                
        except Exception as e:
            print(f"❌ 加载对话失败: {e}")
    
    def clear_conversation(self):
        """清除对话历史"""
        self.conversation_history = []
        print("🗑️  对话历史已清除")
    
    def print_conversation_stats(self):
        """打印对话统计"""
        user_msgs = len([msg for msg in self.conversation_history if msg["role"] == "user"])
        assistant_msgs = len([msg for msg in self.conversation_history if msg["role"] == "assistant"])
        
        print(f"\n📊 对话统计:")
        print(f"   用户消息: {user_msgs} 条")
        print(f"   助手回复: {assistant_msgs} 条")
        print(f"   总消息数: {len(self.conversation_history)} 条")
    
    def start_chat(self):
        """开始聊天会话"""
        print("🎯 开始对话...")
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 你: ").strip()
                
                # 处理特殊命令
                if user_input.lower() in ['退出', 'quit', 'exit']:
                    print("👋 再见！")
                    break
                
                elif user_input.lower() in ['清除', 'clear']:
                    self.clear_conversation()
                    continue
                
                elif user_input.lower() in ['保存', 'save']:
                    self.save_conversation()
                    continue
                
                elif user_input.lower() in ['统计', 'stats']:
                    self.print_conversation_stats()
                    continue
                
                elif user_input.lower() in ['帮助', 'help']:
                    self.print_help()
                    continue
                
                elif user_input.lower() in ['加载', 'load']:
                    filename = input("请输入对话文件路径: ").strip()
                    self.load_conversation(filename)
                    continue
                
                # 空输入
                if not user_input:
                    continue
                
                # 生成回复
                print("🤖 助手思考中...", end="", flush=True)
                response = self.generate_response(user_input)
                print("\r" + " " * 50 + "\r", end="")  # 清除"思考中"提示
                
                # 解析回复
                reasoning, answer = self.extract_reasoning_and_answer(response)
                
                # 打印回复
                print("🤖 助手:")
                if reasoning:
                    print(f"  推理过程: {reasoning}")
                if answer:
                    print(f"  答案: {answer}")
                if not reasoning and not answer:
                    print(f"  {response}")
                
                # 显示对话轮数
                current_turn = len([msg for msg in self.conversation_history if msg["role"] == "user"])
                print(f"  (第 {current_turn} 轮对话)")
                
            except KeyboardInterrupt:
                print("\n\n👋 用户中断，再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                continue
    
    def print_help(self):
        """打印帮助信息"""
        print("\n📖 可用命令:")
        print("  '退出'/'quit' - 结束对话")
        print("  '清除'/'clear' - 清除对话历史")
        print("  '保存'/'save' - 保存对话记录")
        print("  '统计'/'stats' - 显示对话统计")
        print("  '加载'/'load' - 加载对话记录")
        print("  '帮助'/'help' - 显示此帮助信息")
        print("-" * 40)

def select_model():
    """选择模型"""
    models_dir = "./models"
    available_models = []
    
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            model_path = os.path.join(models_dir, item)
            if os.path.isdir(model_path):
                # 检查是否有最佳模型
                best_model_path = os.path.join(model_path, "best_model")
                final_model_path = os.path.join(model_path, "final_model")
                
                if os.path.exists(best_model_path):
                    available_models.append((f"{item}/best_model", best_model_path))
                if os.path.exists(final_model_path):
                    available_models.append((f"{item}/final_model", final_model_path))
                # 如果没有子目录，直接使用模型目录
                elif any(f.endswith('.bin') or f.endswith('.safetensors') for f in os.listdir(model_path)):
                    available_models.append((item, model_path))
    
    if not available_models:
        print("❌ 未找到可用的模型，请指定模型路径")
        return None
    
    print("📁 可用的模型:")
    for i, (name, path) in enumerate(available_models, 1):
        print(f"  {i}. {name}")
    
    try:
        choice = input(f"\n请选择模型 (1-{len(available_models)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(available_models):
            selected_model = available_models[int(choice) - 1][1]
            print(f"✅ 选择模型: {selected_model}")
            return selected_model
        else:
            print("❌ 无效选择")
            return None
    except:
        return None

def main():
    """主函数"""
    print("=" * 60)
    print("🤖 模型对话系统")
    print("=" * 60)
    
    # 选择模型
    model_path = select_model()
    if not model_path:
        # 如果自动选择失败，手动输入路径
        model_path = input("请输入模型路径: ").strip()
        if not os.path.exists(model_path):
            print("❌ 模型路径不存在")
            return
    
    # 初始化聊天机器人
    try:
        chatbot = ChatBot(model_path)
        
        # 检查是否有之前的对话记录
        conversations_dir = "./conversations"
        if os.path.exists(conversations_dir):
            conversation_files = [f for f in os.listdir(conversations_dir) if f.endswith('.json')]
            if conversation_files:
                print(f"\n📂 发现 {len(conversation_files)} 个对话记录")
                load_choice = input("是否加载最近的对话记录? (y/N): ").strip().lower()
                if load_choice == 'y':
                    latest_file = max(conversation_files, key=lambda f: os.path.getctime(os.path.join(conversations_dir, f)))
                    chatbot.load_conversation(os.path.join(conversations_dir, latest_file))
        
        # 开始聊天
        chatbot.start_chat()
        
        # 退出前询问是否保存
        if chatbot.conversation_history:
            save_choice = input("\n是否保存对话记录? (Y/n): ").strip().lower()
            if save_choice != 'n':
                chatbot.save_conversation()
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()