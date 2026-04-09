const chatContainer = document.getElementById('chat-container');
const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

function appendMessage(role, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    msgDiv.innerHTML = `<div class="bubble">${text}</div>`;
    messagesDiv.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return msgDiv;
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage('user', text);
    userInput.value = '';

    // Create assistant message with animated thinking indicator
    const assistantMsg = document.createElement('div');
    assistantMsg.className = 'message assistant';
    assistantMsg.innerHTML = '<div class="bubble"><span class="typing">Devo escribiendo....</span></div>';
    messagesDiv.appendChild(assistantMsg);
    const bubble = assistantMsg.querySelector('.bubble');
    chatContainer.scrollTop = chatContainer.scrollHeight;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        if (!response.ok) throw new Error('Network response was not ok');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullAnswer = '';
        let firstChunk = true;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const dataStr = line.slice(6).trim();
                    if (dataStr === '[DONE]') continue;
                    
                    try {
                        const data = JSON.parse(dataStr);
                        if (data.reset) {
                            fullAnswer = '';
                            bubble.innerHTML = '';
                        }
                        if (data.answer) {
                            if (firstChunk) {
                                bubble.innerHTML = ''; // Clear indicator ONLY when first data arrives
                                firstChunk = false;
                            }
                            fullAnswer += data.answer;
                            bubble.innerText = fullAnswer; 
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        }
                    } catch (e) {
                        console.error('Error parsing stream chunk', e);
                    }
                }
            }
        }
    } catch (error) {
        bubble.innerText = 'Lo siento, no pude conectarme con el servidor.';
        console.error(error);
    }
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});
