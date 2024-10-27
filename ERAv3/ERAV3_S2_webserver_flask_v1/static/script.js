// Get elements
const chatOutput = document.getElementById("chat-output");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");

// Event listener for the send button
sendButton.addEventListener("click", sendMessage);

// Function to send a message
function sendMessage() {
  const message = userInput.value.trim();
  if (message === "") return;

  // Display user's message
  const userMessageElement = document.createElement("div");
  userMessageElement.className = "message user-message";
  userMessageElement.textContent = "You: " + message;
  chatOutput.appendChild(userMessageElement);

  // Clear input
  userInput.value = "";

  // Send message to the server
  fetch("/get_response", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message: message }),
  })
    .then((response) => response.json())
    .then((data) => {
      // Display AI response
      const aiMessageElement = document.createElement("div");
      aiMessageElement.className = "message ai-message";
      aiMessageElement.textContent = data.response;
      chatOutput.appendChild(aiMessageElement);

      // Scroll to the bottom
      chatOutput.scrollTop = chatOutput.scrollHeight;
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}


// Allow pressing Enter to send a message
userInput.addEventListener("keypress", function (e) {
  if (e.key === "Enter") {
    sendMessage();
  }
});
