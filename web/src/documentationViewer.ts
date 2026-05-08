import * as THREE from 'three';

/**
 * Documentation Viewer Modal
 * Displays the ARCHITECTURE_AND_TECHSTACK.md content in a beautiful formatted modal
 */

export class DocumentationViewer {
    private modalElement: HTMLElement | null = null;
    private isOpen: boolean = false;
    private closeButton: HTMLElement | null = null;

    constructor() {
        this.createModal();
        this.attachEventListeners();
    }

    private createModal(): void {
        // Create modal container
        const modal = document.createElement('div');
        modal.id = 'documentation-modal';
        modal.innerHTML = `
            <style>
                #documentation-modal {
                    display: none;
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.85);
                    z-index: 9999;
                    justify-content: center;
                    align-items: center;
                    backdrop-filter: blur(5px);
                    animation: fadeIn 0.3s ease;
                }

                #documentation-modal.open {
                    display: flex;
                }

                @keyframes fadeIn {
                    from {
                        opacity: 0;
                    }
                    to {
                        opacity: 1;
                    }
                }

                .doc-modal-content {
                    background: linear-gradient(135deg, #0a1628 0%, #1a3a52 100%);
                    border: 1px solid rgba(0, 225, 255, 0.3);
                    border-radius: 20px;
                    width: 90%;
                    max-width: 1000px;
                    max-height: 85vh;
                    display: flex;
                    flex-direction: column;
                    box-shadow: 0 0 60px rgba(0, 225, 255, 0.2);
                    animation: slideDown 0.3s ease;
                }

                @keyframes slideDown {
                    from {
                        opacity: 0;
                        transform: translateY(-50px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }

                .doc-header {
                    padding: 30px;
                    border-bottom: 1px solid rgba(0, 225, 255, 0.2);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }

                .doc-title {
                    color: #00e1ff;
                    font-size: 24px;
                    font-weight: 700;
                    margin: 0;
                    text-shadow: 0 0 20px rgba(0, 225, 255, 0.3);
                }

                .doc-close-btn {
                    background: rgba(0, 225, 255, 0.1);
                    border: 2px solid rgba(0, 225, 255, 0.3);
                    color: #00e1ff;
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 24px;
                    transition: all 0.3s ease;
                    font-weight: bold;
                }

                .doc-close-btn:hover {
                    background: rgba(0, 225, 255, 0.2);
                    transform: rotate(90deg);
                }

                .doc-body {
                    overflow-y: auto;
                    padding: 30px;
                    flex: 1;
                    color: #d1d5db;
                    font-size: 14px;
                    line-height: 1.8;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }

                .doc-body::-webkit-scrollbar {
                    width: 8px;
                }

                .doc-body::-webkit-scrollbar-track {
                    background: rgba(0, 225, 255, 0.05);
                }

                .doc-body::-webkit-scrollbar-thumb {
                    background: rgba(0, 225, 255, 0.3);
                    border-radius: 4px;
                }

                .doc-body::-webkit-scrollbar-thumb:hover {
                    background: rgba(0, 225, 255, 0.5);
                }

                .doc-body h1 {
                    color: #00e1ff;
                    font-size: 28px;
                    margin-top: 30px;
                    margin-bottom: 15px;
                    text-shadow: 0 0 15px rgba(0, 225, 255, 0.2);
                }

                .doc-body h2 {
                    color: #64c8ff;
                    font-size: 22px;
                    margin-top: 25px;
                    margin-bottom: 12px;
                }

                .doc-body h3 {
                    color: #a8d9ff;
                    font-size: 18px;
                    margin-top: 20px;
                    margin-bottom: 10px;
                }

                .doc-body h1:first-child {
                    margin-top: 0;
                }

                .doc-body p {
                    margin-bottom: 15px;
                }

                .doc-body code {
                    background: rgba(0, 225, 255, 0.1);
                    color: #00ffcc;
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-family: 'Courier New', monospace;
                    font-size: 13px;
                }

                .doc-body pre {
                    background: rgba(0, 0, 0, 0.4);
                    border: 1px solid rgba(0, 225, 255, 0.2);
                    border-radius: 8px;
                    padding: 15px;
                    overflow-x: auto;
                    margin: 15px 0;
                }

                .doc-body pre code {
                    background: none;
                    color: #00ffcc;
                    padding: 0;
                    font-size: 12px;
                    line-height: 1.6;
                }

                .doc-body ul, .doc-body ol {
                    margin-left: 20px;
                    margin-bottom: 15px;
                }

                .doc-body li {
                    margin-bottom: 8px;
                }

                .doc-body table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                    border: 1px solid rgba(0, 225, 255, 0.2);
                }

                .doc-body th {
                    background: rgba(0, 225, 255, 0.1);
                    color: #00e1ff;
                    padding: 10px;
                    text-align: left;
                    font-weight: 600;
                    border: 1px solid rgba(0, 225, 255, 0.2);
                }

                .doc-body td {
                    padding: 10px;
                    border: 1px solid rgba(0, 225, 255, 0.15);
                }

                .doc-body tr:nth-child(even) {
                    background: rgba(0, 225, 255, 0.05);
                }

                .doc-body a {
                    color: #00e1ff;
                    text-decoration: none;
                    transition: all 0.3s ease;
                }

                .doc-body a:hover {
                    text-decoration: underline;
                    text-shadow: 0 0 10px rgba(0, 225, 255, 0.5);
                }

                .doc-body blockquote {
                    border-left: 4px solid #00e1ff;
                    padding-left: 15px;
                    color: #a8d9ff;
                    margin: 15px 0;
                    font-style: italic;
                }

                /* Responsive */
                @media (max-width: 768px) {
                    .doc-modal-content {
                        width: 95%;
                        max-height: 90vh;
                    }

                    .doc-header {
                        padding: 20px;
                    }

                    .doc-body {
                        padding: 20px;
                    }

                    .doc-title {
                        font-size: 20px;
                    }

                    .doc-body h1 {
                        font-size: 24px;
                    }

                    .doc-body h2 {
                        font-size: 20px;
                    }
                }
            </style>

            <div class="doc-modal-content">
                <div class="doc-header">
                    <h1 class="doc-title">📚 System Documentation</h1>
                    <button class="doc-close-btn" id="doc-close-btn">&times;</button>
                </div>
                <div class="doc-body" id="doc-content">
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        this.modalElement = modal;
        this.closeButton = document.getElementById('doc-close-btn');

        // Load and format documentation
        this.loadDocumentation();
    }

    private async loadDocumentation(): Promise<void> {
        // Documentation panel is intentionally empty
        return;
    }

    private markdownToHtml(markdown: string): string {
        let html = markdown;

        // Escape HTML
        html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

        // Convert markdown
        // Headers
        html = html.replace(/^### (.*?)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.*?)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.*?)$/gm, '<h1>$1</h1>');

        // Bold and italic
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong style="color: #00ffcc;">$1</strong>');
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Code blocks
        html = html.replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>');

        // Inline code
        html = html.replace(/`(.*?)`/g, '<code>$1</code>');

        // Links
        html = html.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank">$1</a>');

        // Lists
        html = html.replace(/^\* (.*?)$/gm, '<li>$1</li>');
        html = html.replace(/^\- (.*?)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*?<\/li>)/s, '<ul>$1</ul>');

        // Horizontal rules
        html = html.replace(/^---$/gm, '<hr style="border: 1px solid rgba(0, 225, 255, 0.2); margin: 30px 0;">');

        // Blockquotes
        html = html.replace(/^> (.*?)$/gm, '<blockquote>$1</blockquote>');

        // Tables (basic support)
        html = html.replace(/\|(.*?)\|/g, '<td>$1</td>');

        // Paragraphs
        html = html.split('\n\n').map(p => p.trim()).filter(p => p && !p.startsWith('<')).map(p => `<p>${p}</p>`).join('');

        return html;
    }

    private attachEventListeners(): void {
        if (this.closeButton) {
            this.closeButton.addEventListener('click', () => this.close());
        }

        // Close on background click
        if (this.modalElement) {
            this.modalElement.addEventListener('click', (e) => {
                if (e.target === this.modalElement) {
                    this.close();
                }
            });
        }

        // Close on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) {
                this.close();
            }
        });
    }

    public open(): void {
        if (this.modalElement) {
            this.modalElement.classList.add('open');
            this.isOpen = true;
            document.body.style.overflow = 'hidden';
        }
    }

    public close(): void {
        if (this.modalElement) {
            this.modalElement.classList.remove('open');
            this.isOpen = false;
            document.body.style.overflow = 'auto';
        }
    }

    public toggle(): void {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }
}
