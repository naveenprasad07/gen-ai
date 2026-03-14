import validators
import streamlit as st
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## Streamlit App
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Sidebar for API Key
with st.sidebar:
    hf_api_key = st.text_input("HuggingFace API Token", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

def summarize_with_hf(text: str, api_key: str) -> str:
    """Directly uses InferenceClient chat_completion — avoids the novita task mismatch."""
    client = InferenceClient(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        token=api_key,
    )
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that summarizes content clearly and concisely."
        },
        {
            "role": "user",
            "content": f"Provide a summary of the following content in 300 words:\n\n{text}"
        }
    ]
    response = client.chat_completion(messages, max_tokens=512)
    return response.choices[0].message.content

if st.button("Summarize the Content from YT or Website"):
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or website URL")
    else:
        try:
            with st.spinner("Loading and summarizing..."):

                ## Load content
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=False
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )

                docs = loader.load()

                ## Combine all doc content into one string
                full_text = "\n\n".join([doc.page_content for doc in docs])

                ## Truncate if too long (LLM token limits)
                if len(full_text) > 12000:
                    full_text = full_text[:12000] + "..."

                ## Summarize directly via InferenceClient
                output_summary = summarize_with_hf(full_text, hf_api_key)

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")