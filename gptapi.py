"""
Install the Google AI Python SDK

$ pip install google-generativeai

See the getting started guide for more information:
https://ai.google.dev/gemini-api/docs/get-started/python
"""

import os

import google.generativeai as genai

genai.configure(api_key="AIzaSyDxy2HKtvwud733EufRiA5DUj3VG6Gx4qU")

# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
]

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash-latest",
  safety_settings=safety_settings,
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "Trong khoảng 5s, yolo đã tìm thấy các hành động sau trong 1 lớp học, hãy sinh ra 1 câu dạng \"\"Hiện tại lớp học đang có {nhãn} bạn đang {nhãn}\" kèm lời nhận xét dựa trên dữ liệu: \"video 1/1 (191/219) /root/yolov5/TestVid.mp4: 3 readings, 2 writings, \"\n\"video 1/1 (192/219) /root/yolov5/TestVid.mp4: 4 readings, 2 writings, \"\n\"video 1/1 (193/219) /root/yolov5/TestVid.mp4: 2 readings, 2 writings, \"\n\"video 1/1 (194/219) /root/yolov5/TestVid.mp4: 3 readings, 2 writings, \"\n\"video 1/1 (195/219) /root/yolov5/TestVid.mp4: 5 sleep, 6 use phone, \"",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Hiện tại lớp học đang có **5 bạn đang ngủ, 6 bạn đang sử dụng điện thoại**.  \n\n**Lời nhận xét:**  Tình hình lớp học có vẻ không tập trung. Số lượng học sinh ngủ và sử dụng điện thoại khá nhiều, điều này có thể ảnh hưởng đến hiệu quả học tập. \n",
      ],
    },
  ]
)


with open("runs/detect/exp/predictions.csv", "r") as file:
    lines = file.readlines()
    print(len(lines))
    message = "Yolo đã tìm thấy các hành động sau trong 1 lớp học, hãy sinh ra 1 câu dạng \"\"Hiện tại lớp học đang có {nhãn} bạn đang {nhãn}\" kèm lời nhận xét dựa trên dữ liệu:"
    for i in range(0, len(lines), 5):
        message += " " + "".join(lines[i:i+5]).replace("\n", "")
        response = chat_session.send_message(message)
        print(response.text)
        with open("response.txt", "a") as file:
            file.write(response.text)
