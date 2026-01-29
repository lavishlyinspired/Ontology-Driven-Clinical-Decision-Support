import ChatInterface from '@/components/ChatInterface'

export const metadata = {
  title: 'Chat - LCA Assistant',
  description: 'Conversational interface for lung cancer clinical decision support'
}

export default function ChatPage() {
  return (
    <div className="chat-fullscreen">
      <ChatInterface />
    </div>
  )
}
