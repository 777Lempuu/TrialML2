@st.cache_resource
def load_model():
    try:
        device = torch.device('cpu')
        
        # Add all required safe globals (for HuggingFace components)
        from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
        from tokenizers import Tokenizer
        torch.serialization.add_safe_globals([BertTokenizerFast, Tokenizer])
        
        # Load with weights_only=False since we've added safe globals
        model_data = torch.load('ag_news_model.pt', 
                             map_location=device,
                             weights_only=False)  # Now safe because we've allowlisted classes
        
        # Recreate model
        model = AutoModelForSequenceClassification.from_pretrained(
            "google/bert_uncased_L-2_H-128_A-2",
            num_labels=4
        )
        model.load_state_dict(model_data['model_state_dict'])
        model.to(device).eval()
        
        tokenizer = model_data['tokenizer']
        st.success("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None, None
