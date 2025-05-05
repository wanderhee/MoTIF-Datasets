# MoTIF: An end-to-end Multimodal Road Traffic Scene Understanding Foundation Model 
The complete dataset will be made publicly available upon publication of the paper.

# :rocket: Main Contributions
Our contributions can be mainly divided into the following parts:
1.  **Fine-tuned MFM**: the research in this paper focuses on improving the recognition ability of MFMs in the complex environment of traffic roads. By means of the low-rank efficient fine-tuning matrix (LoRA), the model proposed in this paper is trained to refine various types of events in road traffic, which ensures the model's accurate recognition and inference ability in road traffic scenarios.
2.  **Multi-metric output and event reasoning for road traffic**: the model proposed in this paper outputs structured descriptive information, which enables the detection and analysis of various types of traffic events. The specific inference ability of the multi-modal large model makes it possible to further deduce the traffic state.
3.  **Video annotated dataset for traffic roads**: In this paper, we construct a multimodal exclusive dataset for video detection of traffic roads. We propose a set of automated video image annotation methods for traffic target object detection and semantic segmentation fusion for highways, and use Q-A question and answer pairs to annotate roadside surveillance videos of highways. Based on the above video and annotation information, this paper constructs an industry standard multi-modal dataset.

# :collision: Sample Demonstration
Here is an example from the dataset our research team has constructed – a sample sourced from The Tianjin Road Intersection Dataset.

https://private-user-images.githubusercontent.com/182067692/437971419-d7a61d2a-c359-40c8-98e0-0e268f1dbfcb.mp4

The annotations for the aforementioned example video are structured as follows: The annotation files consist of a list of dictionaries, with each entry adhering to the uniform format below:

```
[
    {
        "id": 0,
        "video": "datasets/example.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "<video>\nClarify the contents of the displayed video with great detail."
            },
            {
                "from": "gpt",
                "value": "First, two black cars drove out of the intersection from bottom to top, and one black car drove out of the intersection from top to bottom. Next, the traffic lights turned from green to red, and a white car stopped outside the intersection at the bottom of the video. Then, multiple electric cars and cars drove through the intersection in an orderly manner. Finally, a black car turned left."
            }
        ]
    },
    ...
]
```
