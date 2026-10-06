# Modified by the TACD authors from the Tencent HY-Motion-1.0 Space (hymotion/utils/gradio_css.py): header, footer and placeholder text.
import os.path as osp

# create interface
APP_CSS = """
    :root{
    --primary-start:#667eea; --primary-end:#764ba2;
    --secondary-start:#4facfe; --secondary-end:#00f2fe;
    --accent-start:#f093fb; --accent-end:#f5576c;
    --page-bg:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);
    --card-bg:linear-gradient(135deg,#ffffff 0%,#f8f9fa 100%);
    --radius:12px;
    --iframe-bg:#ffffff;
    }

    /* Dark mode variables */
    [data-theme="dark"], .dark {
    --page-bg:linear-gradient(135deg,#1a1a1a 0%,#2d3748 100%);
    --card-bg:linear-gradient(135deg,#2d3748 0%,#374151 100%);
    --text-primary:#f7fafc;
    --text-secondary:#e2e8f0;
    --border-color:#4a5568;
    --input-bg:#374151;
    --input-border:#4a5568;
    --iframe-bg:#1a1a2e;
    }

    /* Page and card */
    .gradio-container{
    background:var(--page-bg) !important;
    min-height:100vh !important;
    color:var(--text-primary, #333) !important;
    }

    .main-header{
    background:transparent !important; border:none !important; box-shadow:none !important;
    padding:0 !important; margin:10px 0 16px !important;
    text-align:center !important;
    }

    .main-header h1, .main-header p, .main-header li {
    color:var(--text-primary, #333) !important;
    }

    .left-panel,.right-panel{
    background:var(--card-bg) !important;
    border:1px solid var(--border-color, #e9ecef) !important;
    border-radius:15px !important;
    box-shadow:0 4px 20px rgba(0,0,0,.08) !important;
    padding:24px !important;
    }

    .gradio-accordion{
    border:1px solid var(--border-color, #e1e5e9) !important;
    border-radius:var(--radius) !important;
    margin:12px 0 !important; background:transparent !important;
    }

    .gradio-accordion summary{
    background:transparent !important;
    padding:14px 18px !important;
    font-weight:600 !important;
    color:var(--text-primary, #495057) !important;
    }

    .gradio-group{
    background:transparent !important; border:none !important;
    border-radius:8px !important; padding:12px 0 !important; margin:8px 0 !important;
    }

    /* Input class style - dark mode adaptation */
    .gradio-textbox input,.gradio-textbox textarea,.gradio-dropdown .wrap{
    border-radius:8px !important;
    border:2px solid var(--input-border, #e9ecef) !important;
    background:var(--input-bg, #fff) !important;
    color:var(--text-primary, #333) !important;
    transition:.2s all !important;
    }

    .gradio-textbox input:focus,.gradio-textbox textarea:focus,.gradio-dropdown .wrap:focus-within{
    border-color:var(--primary-start) !important;
    box-shadow:0 0 0 3px rgba(102,126,234,.1) !important;
    }

    .gradio-slider input[type="range"]{
    background:linear-gradient(to right,var(--primary-start),var(--primary-end)) !important;
    border-radius:10px !important;
    }

    .gradio-checkbox input[type="checkbox"]{
    border-radius:4px !important;
    border:2px solid var(--input-border, #e9ecef) !important;
    transition:.2s all !important;
    }

    .gradio-checkbox input[type="checkbox"]:checked{
    background:linear-gradient(45deg,var(--primary-start),var(--primary-end)) !important;
    border-color:var(--primary-start) !important;
    }

    /* Label text color adaptation */
    .gradio-textbox label, .gradio-dropdown label, .gradio-slider label,
    .gradio-checkbox label, .gradio-html label {
    color:var(--text-primary, #333) !important;
    }

    .gradio-textbox .info, .gradio-dropdown .info, .gradio-slider .info,
    .gradio-checkbox .info {
    color:var(--text-secondary, #666) !important;
    }

    /* Status information - dark mode adaptation */
    .gradio-textbox[data-testid*="状态信息"] input{
    background:var(--input-bg, linear-gradient(135deg,#f8f9fa 0%,#e9ecef 100%)) !important;
    border:2px solid var(--input-border, #dee2e6) !important;
    color:var(--text-primary, #495057) !important;
    font-weight:500 !important;
    }

    /* Status textbox - dynamic height based on content */
    .status-textbox textarea{
    height:auto !important;
    min-height:2.5em !important;
    resize:none !important;
    overflow-y:hidden !important;
    }

    /* Button base class and variant */
    .generate-button,.rewrite-button,.dice-button{
    border:none !important; color:#fff !important; font-weight:600 !important;
    border-radius:8px !important; transition:.3s all !important;
    box-shadow:0 4px 15px rgba(0,0,0,.12) !important;
    }

    .generate-button{ background:linear-gradient(45deg,var(--primary-start),var(--primary-end)) !important; }
    .rewrite-button{ background:linear-gradient(45deg,var(--secondary-start),var(--secondary-end)) !important; }
    .dice-button{
    background:linear-gradient(45deg,var(--accent-start),var(--accent-end)) !important;
    height:40px !important;
    }

    .generate-button:hover,.rewrite-button:hover{ transform:translateY(-2px) !important; }
    .dice-button:hover{
    transform:scale(1.05) !important;
    box-shadow:0 4px 12px rgba(240,147,251,.28) !important;
    }

    .dice-container{
    display:flex !important;
    align-items:flex-end !important;
    justify-content:center !important;
    }

    /* Right panel clipping overflow, avoid double scrollbars */
    .right-panel{
    background:var(--card-bg) !important;
    border:1px solid var(--border-color, #e9ecef) !important;
    border-radius:15px !important;
    box-shadow:0 4px 20px rgba(0,0,0,.08) !important;
    padding:24px !important; overflow:hidden !important;
    }

    /* Main content row - ensure equal heights */
    .main-row {
    display: flex !important;
    align-items: stretch !important;
    }

    /* Flask area - match left panel height */
    .flask-display{
    padding:0 !important; margin:0 !important; border:none !important;
    box-shadow:none !important; background:var(--iframe-bg) !important;
    border-radius:10px !important; position:relative !important;
    height:100% !important; min-height:750px !important;
    display:flex !important; flex-direction:column !important;
    }

    .flask-display iframe{
    width:100% !important; flex:1 !important; min-height:750px !important;
    border:none !important; border-radius:10px !important; display:block !important;
    background:var(--iframe-bg) !important;
    }

    /* Right panel should stretch to match left panel */
    .right-panel{
    background:var(--card-bg) !important;
    border:1px solid var(--border-color, #e9ecef) !important;
    border-radius:15px !important;
    box-shadow:0 4px 20px rgba(0,0,0,.08) !important;
    padding:24px !important; overflow:hidden !important;
    display:flex !important; flex-direction:column !important;
    }

    /* Ensure dropdown menu is visible in dark mode */
    [data-theme="dark"] .gradio-dropdown .wrap,
    .dark .gradio-dropdown .wrap {
    background:var(--input-bg) !important;
    color:var(--text-primary) !important;
    }

    [data-theme="dark"] .gradio-dropdown .option,
    .dark .gradio-dropdown .option {
    background:var(--input-bg) !important;
    color:var(--text-primary) !important;
    }

    [data-theme="dark"] .gradio-dropdown .option:hover,
    .dark .gradio-dropdown .option:hover {
    background:var(--border-color) !important;
    }

    .footer{
    text-align:center !important;
    margin-top:20px !important;
    padding:10px !important;
    color:var(--text-secondary, #666) !important;
    }

    /* Example Gallery Styles */
    .example-gallery-display{
    padding:0 !important; margin:12px 0 !important; border:none !important;
    box-shadow:none !important; background:var(--iframe-bg) !important;
    border-radius:10px !important; position:relative !important;
    min-height:500px !important;
    }

    .example-gallery-display iframe{
    width:100% !important; min-height:500px !important;
    border:none !important; border-radius:10px !important; display:block !important;
    background:var(--iframe-bg) !important;
    }
"""

HEADER_BASE_MD = """# TACD: Text-to-Motion in 8 Steps
### [Paper](https://arxiv.org/abs/2610.02867) · [Project page](https://vkgo.github.io/TACD/) · [Code](https://github.com/vkgo/TACD) · [Model](https://huggingface.co/weijinhuang/TACD-HY-Motion-Lite)"""""

FOOTER_MD = """Tencent HY-MOTION 1.0 is licensed under the Tencent HY-MOTION 1.0 Community License Agreement,
Copyright © 2025 Tencent. All Rights Reserved. The trademark rights of “Tencent HY” are owned by Tencent or its affiliate.
Powered by Tencent HY."""

# Path to placeholder scene HTML template
PLACEHOLDER_SCENE_TEMPLATE = osp.join(osp.dirname(__file__), "..", "..", "scripts/gradio/templates/placeholder_scene.html")



def get_placeholder_html() -> str:
    """
    Load the placeholder scene HTML and wrap it in an iframe for display.
    Returns an iframe HTML string with the embedded placeholder scene.
    """
    try:
        with open(PLACEHOLDER_SCENE_TEMPLATE, "r", encoding="utf-8") as f:
            html_content = f.read()
        # Escape HTML content for srcdoc attribute
        escaped_html = html_content.replace('"', '&quot;')
        iframe_html = f'''
            <iframe
                srcdoc="{escaped_html}"
                width="100%"
                height="750px"
                style="border: none; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);"
            ></iframe>
        '''
        return iframe_html
    except Exception as e:
        print(f">>> Failed to load placeholder scene HTML: {e}")
        # Fallback to simple placeholder
        return """
        <div style='height: 750px; width: 100%; border-radius: 8px; border-color: #e5e7eb; border-style: solid; border-width: 1px; display: flex; justify-content: center; align-items: center; background: #424242;'>
            <div style='text-align: center; font-size: 16px; color: #a0aec0;'>
                <p>Welcome to TACD!</p>
                <p>Enter a text description and generate motion to see the 3D visualization here.</p>
            </div>
        </div>
        """


WITHOUT_PROMPT_ENGINEERING_WARNING = """
    <div style='color: #ff0000; font-weight: bold;'>
        <p>Prompt engineering is not available. You should use `A person ...` format to describe the motion and manually adjust the duration. Click [📚 Example Prompts] to see more examples.</p>
        <p>Non-humanoid Characters, Multi-person Interactions and Environment & Camera are not supported.</p>
    </div>
"""