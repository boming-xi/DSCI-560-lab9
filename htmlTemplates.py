css = """
<style>

/* expand main page width */
.block-container {
    max-width: 1300px;
    padding-top: 2rem;
}

/* chat message container */
.chat-message {
    padding: 1.5rem;
    border-radius: 0.7rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
}

/* user message background */
.chat-message.user {
    background-color: #2b313e;
}

/* bot message background */
.chat-message.bot {
    background-color: #475063;
}

/* avatar section */
.chat-message .avatar {
  width: 60px;
}

/* avatar image */
.chat-message .avatar img {
  max-width: 60px;
  max-height: 60px;
  border-radius: 50%;
  object-fit: cover;
}

/* message text */
.chat-message .message {
  width: 100%;
  padding-left: 1rem;
  color: #ffffff;
  font-size: 16px;
  line-height: 1.5;
}

</style>
"""


bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""


user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/1144/1144760.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""