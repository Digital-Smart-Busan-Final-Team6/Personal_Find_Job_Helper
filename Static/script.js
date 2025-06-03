// static/script.js

// 폼 내부에서 hidden CSRF 토큰 값을 읽어오는 헬퍼 함수
function getCsrfToken() {
    const tokenInput = document.querySelector('input[name=csrfmiddlewaretoken]');
    return tokenInput ? tokenInput.value : '';
}

// AJAX POST 요청 함수 예시
async function sendChatMessage(message) {
    const csrftoken = getCsrfToken();
    try {
        const response = await fetch("/chat/", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-CSRFToken': csrftoken,
                'X-Requested-With': 'XMLHttpRequest',
            },
            body: new URLSearchParams({ message })
        });

        if (!response.ok) {
            console.error("서버 오류:", response.status);
            return null;
        }
        const data = await response.json();
        return data.bot;
    } catch (err) {
        console.error("네트워크 오류:", err);
        return null;
    }
}

// 페이지 로드 후 이벤트 연결
document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatBox = document.getElementById('chat-box');

    // 채팅 창 맨 아래로 스크롤
    function scrollChatToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // 메시지 화면에 추가
    function appendMessage(text, isUser) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('chat-message', isUser ? 'user' : 'bot');
        msgDiv.textContent = text;
        chatBox.appendChild(msgDiv);
        scrollChatToBottom();
    }

    scrollChatToBottom();

    if (!chatForm || !chatInput) return;

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = chatInput.value.trim();
        if (!message) return;

        // 1) 사용자 메시지 화면에 띄우기
        appendMessage(message, true);
        chatInput.value = '';
        chatInput.focus();

        // 2) AJAX로 서버에 메시지 보내고 응답받기
        const botReply = await sendChatMessage(message);
        if (botReply !== null) {
            appendMessage(botReply, false);
        } else {
            appendMessage("챗봇 응답을 받을 수 없습니다.", false);
        }
    });
});
