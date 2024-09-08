from utils import *
import gradio as gr
gr.ChatInterface(predict).launch(debug=True, share=True)