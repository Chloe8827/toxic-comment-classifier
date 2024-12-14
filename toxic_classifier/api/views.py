# Import necessary libraries
from rest_framework.views import APIView  # REST framework view class for API endpoints
from rest_framework.response import Response  # Class for handling API responses
from transformers import pipeline  # Hugging Face transformers pipeline for model inference
from peft import PeftModel  # Parameter-Efficient Fine-Tuning model implementation

class ModelManager:
    """
    Model Manager Class
    
    Responsible for loading and managing all classification models.
    Handles model initialization, tokenization, and inference operations.
    Supports multiple classification labels with individual models for each.
    """
    
    def __init__(self):
        """
        Initialize the Model Manager
        
        Sets up the computing device (GPU/CPU) and initializes storage dictionaries
        for models, tokenizers, and classifiers. Loads pre-trained models for each
        classification label.
        """
        # Set device based on GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize storage dictionaries for components
        self.models = {}      # Dictionary to store loaded models
        self.tokenizers = {}  # Dictionary to store tokenizers for each model
        self.classifiers = {} # Dictionary to store classification pipelines
        
        # Load models for each classification label
        for label in LABELS:
            try:
                # Construct path to the model files
                model_path = os.path.join(MODEL_PATH, f'basic_lora_xlm_{label}')
                
                # Initialize tokenizer using XLM-RoBERTa base model
                self.tokenizers[label] = AutoTokenizer.from_pretrained(
                    'FacebookAI/xlm-roberta-base'
                )
                
                # Load the base model for binary classification
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    'FacebookAI/xlm-roberta-base',
                    num_labels=2  # Binary classification setup
                )
                
                # Load and apply LoRA weights to the base model
                config = PeftConfig.from_pretrained(model_path)
                model = PeftModel.from_pretrained(base_model, model_path)
                model.to(self.device)  # Move model to appropriate device
                
                # Set up the classification pipeline
                self.classifiers[label] = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=self.tokenizers[label],
                    return_all_scores=True  # Return probabilities for all classes
                )
            
    def predict(self, text):
        """
        Perform multi-label prediction on input text
        
        Args:
            text (str): Input text to be classified
            
        Returns:
            dict: Dictionary containing prediction scores for each label
                  Format: {label_name: probability_score}
                  
        Notes:
            - Processes the text through each label's classifier
            - Returns probability scores for the positive class (LABEL_1)
            - Handles cases where a classifier might be missing for a label
        """
        results = {}
        for label in LABELS:
            if label in self.classifiers:
                # Get predictions from the classifier
                prediction = self.classifiers[label](text)
                scores = prediction[0]
                
                # Extract the positive class probability
                positive_score = next(
                    (score['score'] for score in scores if score['label'] == 'LABEL_1'),
                    scores[1]['score'] if len(scores) > 1 else scores[0]['score']
                )
                
                # Store the result
                results[label] = float(positive_score)
                
        return results