from main import TextAnalyzer, AppConfig
import gradio as gr

class GradioInterface:
    def __init__(self, analyzer: TextAnalyzer):
        self.analyzer = analyzer
    
    def create_interface(self):
        return gr.Interface(
            fn=self._gradio_predict,
            inputs=gr.Textbox(lines=3, placeholder="Enter text to analyze..."),
            outputs=gr.Label(num_top_classes=2),
            title="Snowflake Sensitivity Classifier+",
            description="Enhanced classifier with zero-shot validation",
            examples=[
                ["That's hilarious! Dark humor at its finest!"],
                ["People like you shouldn't be allowed to vote"],
                ["This joke might be too edgy for some audiences"],
                ["You should not vote because you are a woman"]
            ]
        )
    
    def _gradio_predict(self, text: str):
        analysis = self.analyzer.analyze_text(text)
        return {
            "offensive": analysis['combined_offensive'],
            "safe_for_snowflake": analysis['combined_safe']
        }


if __name__ == '__main__':
  config = AppConfig()
  analyzer = TextAnalyzer(config)
  
  gradio_interface = GradioInterface(analyzer)
  gradio_interface.create_interface().launch()