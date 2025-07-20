import sys
import json
from tqdm import tqdm
from typing import List, Dict

sys.path.append('./')
from motif import model_init, mm_infer
from motif.utils import disable_torch_init

# system prompt
SYSTEM_PROMPT = ""

#固定提问模板（无需修改）
USER_INSTRUCT = """你是一名高速公路的交通管理人员。
输入视频为交通正常运行的情况或以下五种类型的交通事件之一：交通事故、交通拥堵、道路施工、异常停车以及行人闯入。\n
您的任务是确定视频内容中包含的交通事件类型，用数字标记结果并给出交通事件的描述。以下是每个交通事件类型的定义，如果存在以下情况则说明可能发生了对应事件：\n
（1） 正常情况：视频中车辆和行人的速度、方向或距离没有异常变化，保持连续稳定的运动轨迹，没有检测到异常物体。\n
（2） 交通事故：在视频中，发生了车辆碰撞、车辆与护栏碰撞、车辆倾覆或火灾等事件。这些事件通常伴随着速度、方向或距离的突然变化，导致运动轨迹的突然中断。\n
（3） 交通拥堵：在视频中，车辆逐渐聚集形成队列，以低速行驶或完全静止。这些事件通常伴随着交通密度的逐渐增加。\n
（4） 道路施工：视频中，道路上有施工标志（如锥体）、施工车辆（施工车辆、压路机等）或施工人员（穿着反光背心工作）、材料堆（沥青/钢板等），施工人员正在进行道路维护、维修或重建活动。这些事件通常伴随着车道封闭、警告标志和锥形路标等特征；\n
（5） 异常停车：在视频中，存在车辆在高速公路上无故突然停车或因故障等原因停在行车道或应急车道，这通常伴随着车辆的突然减速或停止，可能引起后方车辆的紧急制动或变道；\n
（6） 行人闯入：在视频中，存在行人非法进入高速公路或在高速公路上行走。\n
我们将用数字标记每个类别：1表示“正常情况”，2表示“交通事故”，3表示“交通拥堵”，4表示“道路施工”，5表示“异常停车”，6表示“行人闯入”。输入视频仅描绘了六种情况的其中之一。请根据视频内容选择最合适的标签，并且给出交通事件的描述。请按照以下示例格式进行回答，不需要回答其他不相关内容:
交通事件：检测到的事件的编号\n
事件描述：对该交通事件的描述（如事件发生过程、天气状况、对交通造成的影响、周围环境情况以及排队情况等）\n"""

def format_output(test_data: List[Dict], predictions: List[str]) -> Dict:
    """构建符合要求的输出结构（保持原有格式）"""
    return {
        "metadata": {
            "model": "videollama2",
            "dataset": "datasets_intersection",
            "eval_tools": ["BLEU-4", "ROUGE-L", "CIDEr", "BERTScore"]
        },
        "results": [
            {
                "id": idx+1,
                "input": {
                    "video": item["video"],
                    "reference_text": item["reference"]
                },
                "output": {
                    "generated_text": pred
                },
                "evaluation_scores": initialize_metrics()
            }
            for idx, (item, pred) in enumerate(zip(test_data, predictions))
        ],
        "aggregated_scores": initialize_aggregated_scores()
    }

def initialize_metrics() -> Dict:
    """初始化指标数据结构（保持不变）"""
    return {
        "text_metrics": {
            "BLEU": {"bleu_4": 0.0},
            "ROUGE": {"rouge_l": {"f": 0.0}},
            "CIDEr": {"cider": 0.0},
            "BERTScore": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }
        }
    }

def initialize_aggregated_scores() -> Dict:
    """初始化汇总指标（保持不变）"""
    return {
        "text_metrics": {
            "BLEU-4_mean": 0.0,
            "ROUGE-L_f1_mean": 0.0,
            "CIDEr_mean": 0.0,
            "BERTScore_f1_mean": 0.0
        }
    }

def batch_inference(test_json_path: str, output_json_path: str) -> None:
    # 初始化设置
    disable_torch_init()
    
    # 加载测试数据
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    
    # 模型初始化
    model, processor, tokenizer = model_init('../VideoLLaMA2.1-7B-16F')
    video_processor = processor['video']
    
    # 执行推理
    predictions = []
    for item in tqdm(test_data, desc="Analyzing Intersections"):
        try:
            # 统一使用固定提问模板
            full_prompt = f"System: {SYSTEM_PROMPT}\n\nUser: {USER_INSTRUCT}"
            video_path = f"datasets_highway/custom_sft/{item['video']}"
            visual_inputs = video_processor(video_path)
            pred = mm_infer(
                visual_inputs,
                full_prompt,  # 使用统一提问模板
                model=model,
                tokenizer=tokenizer,
                do_sample=False,
                modal='video'
            ).strip()
            
            # 清理多余换行符
            pred = " ".join(pred.split()) 
            
        except Exception as e:
            pred = f"ERROR: {str(e)}"
        predictions.append(pred)
    
    # 构建输出结构
    output_data = format_output(test_data, predictions)
    
    # 保存结果
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # 配置路径
    input_json = "datasets_highway/custom_sft/classify.json"
    output_json = "results/test_novp_vl2_classify_outouts.json"
    
    # 执行推理
    batch_inference(input_json, output_json)