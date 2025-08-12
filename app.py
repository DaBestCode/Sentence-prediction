import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

def generate_text_with_temperature(generator, prompt, max_length=100, temperature=1.0, top_p=0.9, top_k=50):
    """Generate text using the Hugging Face pipeline with sampling parameters."""
    
    result = generator(
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id,
        return_full_text=False
    )
    
    return result[0]['generated_text']


@st.cache_resource
def load_model_and_tokenizer(model_path):
    """Load the fine-tuned model and tokenizer using the pipeline."""
    try:
        # Load the model directly from the Hugging Face Hub
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)

        generator = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device=-1  # Use CPU for a general purpose setup
        )

        st.success("✅ Model and tokenizer loaded successfully from Hugging Face Hub!")
        
        return generator
    
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {str(e)}")
        st.error("Please ensure the model path is correct and the repository is public or you are authenticated.")
        return None

def main():
    st.title("🤖 Fine-tuned GPT-2 Text Generator")
    st.write("Generate text using your fine-tuned GPT-2 model with adjustable temperature and sampling parameters.")
    
    model_path = st.text_input(
        "Hugging Face Model Hub Path:", 
        value="Pruthvi-1029/fine-tuned-distilgpt2",
        help="Path to your saved fine-tuned model on Hugging Face Hub"
    )
    
    if model_path:
        with st.spinner("Loading model..."):
            generator = load_model_and_tokenizer(model_path)
        
        if generator is not None:
            st.header("Text Generation")
            
            prompt = st.text_area(
                "Enter your prompt:",
                value="Artificial intelligence will",
                height=100,
                help="Start typing your prompt here. The model will continue from where you leave off."
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.7,
                    step=0.1,
                    help="Controls randomness. Higher = more creative, Lower = more focused"
                )
                
                max_length = st.slider(
                    "Max Length",
                    min_value=10,
                    max_value=200,
                    value=100,
                    step=10,
                    help="Maximum number of tokens to generate"
                )
            
            with col2:
                top_p = st.slider(
                    "Top-p (Nucleus)",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.9,
                    step=0.05,
                    help="Nucleus sampling. Only tokens with cumulative probability up to p are considered"
                )
                
                top_k = st.slider(
                    "Top-k",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=5,
                    help="Only consider the top k most likely tokens. Set to 0 to disable"
                )
            
            if st.button("🚀 Generate Text", type="primary"):
                if prompt.strip():
                    with st.spinner("Generating text..."):
                        try:
                            generated_text = generate_text_with_temperature(
                                generator=generator,
                                prompt=prompt,
                                max_length=max_length,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k
                            )
                            
                            st.header("Generated Text")
                            st.write("**Prompt + Generated continuation:**")
                            st.text_area("", value=prompt + generated_text, height=200, disabled=True)
                            
                            st.write("**Generated portion only:**")
                            st.text_area("", value=generated_text, height=150, disabled=True)
                            
                        except Exception as e:
                            st.error(f"Error during generation: {str(e)}")
                else:
                    st.warning("Please enter a prompt!")
            
            with st.expander("ℹ️ Model Information"):
                st.write(f"**Model Path:** {model_path}")
                st.write(f"**Tokenizer:** {generator.tokenizer.__class__.__name__}")
                st.write(f"**Model:** {generator.model.__class__.__name__}")
        
        else:
            st.error("Failed to load model. Please check the model path and ensure all required files are present.")
    
    with st.sidebar:
        st.header("💡 Generation Tips")
        st.write("""
        **Temperature:**
        - 0.0: Deterministic (always picks most likely token)
        - 0.7: Balanced creativity and coherence
        - 1.0+: Very creative but potentially incoherent
        
        **Top-p (Nucleus):**
        - 0.9: Good balance
        - Lower: More focused
        - Higher: More diverse
        
        **Top-k:**
        - 50: Good default
        - Lower: More focused
        - Higher: More diverse
        - 0: Disabled
        """)
        
        st.header("📝 Model Details")
        st.write("""
        This app loads your fine-tuned GPT-2 model directly from the Hugging Face Hub, which is the recommended way to handle large models for deployment.
        """)

if __name__ == "__main__":
    main()