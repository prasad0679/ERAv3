import gradio as gr
from openai import OpenAI
import os
import json

def load_api_key():
    with open('api_key.txt', 'r') as file:
        return file.read().strip()

def load_prompt_template(prompt_file):
    with open(prompt_file, 'r') as file:
        return file.read()

def process_with_openai(email_content, prompt_file):
    try:
        # Load API key and create client
        api_key = load_api_key()
        client = OpenAI(api_key=api_key)

        # Load and format prompt
        prompt_template = load_prompt_template(prompt_file)
        formatted_prompt = prompt_template.format(email_content=email_content)

        # Make API call to OpenAI using new client syntax
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant. Return only valid JSON without any additional text or formatting."
                },
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.1  # Lower temperature for more consistent JSON output
        )

        # Get the response content and clean it
        response_content = response.choices[0].message.content.strip()
        
        # Try to parse the response as JSON
        try:
            # Remove any leading/trailing whitespace or quotes
            cleaned_response = response_content.strip().strip('"').strip()
            
            # If response doesn't start with {, try to find and extract the JSON object
            if not cleaned_response.startswith("{"):
                start_idx = cleaned_response.find("{")
                if start_idx != -1:
                    cleaned_response = cleaned_response[start_idx:]
            
            # If response doesn't end with }, try to find and extract the JSON object
            if not cleaned_response.endswith("}"):
                end_idx = cleaned_response.rfind("}")
                if end_idx != -1:
                    cleaned_response = cleaned_response[:end_idx+1]
            
            # Parse the cleaned response as JSON
            json_response = json.loads(cleaned_response)
            
            # Convert back to a formatted string for display
            return json.dumps(json_response, indent=2, ensure_ascii=False)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON response received:\n{response_content}\nError details: {str(e)}"

    except Exception as e:
        return f"An error occurred: {str(e)}"

def extract_customer_concern(email_content):
    return process_with_openai(email_content, 'prompt_ExtractCustomerConcern.txt')

def extract_customer_data(email_content):
    return process_with_openai(email_content, 'prompt_ExtractCustomerData.txt')

def check_data_completeness(email_content):
    try:
        # First extract the customer data
        extracted_data = process_with_openai(email_content, 'prompt_ExtractCustomerData.txt')
        
        # Parse the extracted data
        if isinstance(extracted_data, str) and extracted_data.startswith("Error"):
            return f"Data extraction failed: {extracted_data}"
            
        try:
            data_json = json.loads(extracted_data)
        except json.JSONDecodeError:
            return f"Failed to parse extracted data: {extracted_data}"
        
        # Now check completeness using both email and extracted data
        prompt_template = load_prompt_template('prompt_CheckData.txt')
        formatted_prompt = prompt_template.format(
            email_content=email_content,
            extracted_data=json.dumps(data_json, indent=2)
        )
        
        # Make API call for completeness check
        api_key = load_api_key()
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful customer service assistant. Analyze the data completeness and respond with valid JSON in the format: {\"complete\": \"yes/no\", \"missing\": []}"
                },
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.1
        )
        
        completeness_response = response.choices[0].message.content.strip()
        
        # Try to parse the response, if it fails, try to clean it up
        try:
            json_response = json.loads(completeness_response)
        except json.JSONDecodeError:
            # Try to clean up the response
            cleaned_response = completeness_response.replace("'", "\"").strip()
            if not cleaned_response.startswith("{"):
                cleaned_response = "{" + cleaned_response.split("{", 1)[1]
            if not cleaned_response.endswith("}"):
                cleaned_response = cleaned_response.rsplit("}", 1)[0] + "}"
            try:
                json_response = json.loads(cleaned_response)
            except json.JSONDecodeError:
                return f"Invalid response format received:\n{completeness_response}"
        
        return json.dumps(json_response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        return f"An error occurred during completeness check: {str(e)}"

# Create Gradio interface
def create_interface():
    # Define theme
    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=["Helvetica", "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
        button_primary_text_color="white",
        block_label_text_size="sm",
        block_title_text_size="xl",
    )

    # Custom CSS for styling
    custom_css = """
        .container {
            max-width: 800px !important;
            margin: auto !important;
            padding: 20px !important;
        }
        .title {
            text-align: center !important;
            color: #2C3E50 !important;
            margin-bottom: 2rem !important;
            font-weight: bold !important;
            font-size: 2.5rem !important;
        }
        .button-row {
            display: flex !important;
            justify-content: space-between !important;
            gap: 10px !important;
            margin: 20px 0 !important;
        }
        .input-box, .output-box {
            border-radius: 8px !important;
            border: 1px solid #E5E7EB !important;
        }
        .input-box:focus, .output-box:focus {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
        }
    """

    with gr.Blocks(theme=theme, css=custom_css) as demo:
        with gr.Column(elem_classes="container"):
            gr.Markdown(
                "# Email Analysis Tool", 
                elem_classes="title"
            )
            
            email_input = gr.Textbox(
                label="Customer Email",
                placeholder="Paste customer email here...",
                lines=5,
                elem_classes="input-box",
                scale=1
            )

            with gr.Row(elem_classes="button-row"):
                extract_concern_button = gr.Button(
                    "Extract Customer Concern",
                    variant="primary",
                    scale=1
                )
                extract_data_button = gr.Button(
                    "Extract Customer Data",
                    variant="primary",
                    scale=1
                )
                check_data_button = gr.Button(
                    "Check Data Completeness",
                    variant="primary",
                    scale=1
                )

            output = gr.Textbox(
                label="Analysis Results",
                lines=8,
                elem_classes="output-box",
                scale=1
            )

            # Connect buttons to their respective functions
            extract_concern_button.click(
                fn=extract_customer_concern,
                inputs=[email_input],
                outputs=[output]
            )
            
            extract_data_button.click(
                fn=extract_customer_data,
                inputs=[email_input],
                outputs=[output]
            )
            
            check_data_button.click(
                fn=check_data_completeness,
                inputs=[email_input],
                outputs=[output]
            )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_api=False
    ) 