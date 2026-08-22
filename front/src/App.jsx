import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import ChatPage from "./pages/ChatPage/chat";

function App() {

  return (
    <Router>
      <Routes>
        <Route path="/chat" element={<ChatPage/>}/>
      </Routes>
    </Router>
  )
}

export default App