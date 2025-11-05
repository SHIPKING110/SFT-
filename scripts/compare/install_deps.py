# scripts/install_deps.py
import subprocess
import sys

def install_packages():
    """安装评估脚本所需的依赖包"""
    packages = [
        "evaluate",
        "rouge-score", 
        "nltk",
        "absl-py",
        "pandas",
        "numpy",
        "datasets"
    ]
    
    print("正在安装评估依赖...")
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ 成功安装: {package}")
        except subprocess.CalledProcessError:
            print(f"❌ 安装失败: {package}")
        except Exception as e:
            print(f"⚠️  安装 {package} 时出错: {e}")

if __name__ == "__main__":
    install_packages()