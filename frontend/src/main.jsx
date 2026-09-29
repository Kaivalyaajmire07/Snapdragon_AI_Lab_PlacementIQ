import React, {useEffect, useMemo, useState} from "react";
import {createRoot} from "react-dom/client";
import {LayoutDashboard, BrainCircuit, Search, LogOut, Cpu, ShieldCheck, RefreshCw} from "lucide-react";
import {BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer} from "recharts";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function api(path, options={}) {
  const token = localStorage.getItem("placementiq_token");
  const headers = {...(options.headers||{})};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (options.body && !(options.body instanceof FormData)) headers["Content-Type"]="application/json";
  const r = await fetch(API+path,{...options,headers});
  const data = await r.json().catch(()=>({}));
  if (!r.ok) throw new Error(data.detail || "Request failed");
  return data;
}

function Login({onLogin}) {
  const [username,setUsername]=useState("faculty");
  const [password,setPassword]=useState("faculty123");
  const [busy,setBusy]=useState(false), [error,setError]=useState("");
  async function submit(e){
    e.preventDefault(); setBusy(true); setError("");
    try { const d=await api("/auth/login",{method:"POST",body:JSON.stringify({username,password})});
      localStorage.setItem("placementiq_token",d.access_token); onLogin(d.user);
    } catch(e){setError(e.message)} finally{setBusy(false)}
  }
  return <div className="login-shell"><div className="login-card">
    <div className="brand"><Cpu size={30}/><div><b>PlacementIQ</b><span>Snapdragon AI Lab Build</span></div></div>
    <h1>Placement Intelligence</h1><p className="muted">AI-powered placement analytics designed for local AI-PC deployment.</p>
    <form onSubmit={submit}><label>Username<input value={username} onChange={e=>setUsername(e.target.value)}/></label>
    <label>Password<input type="password" value={password} onChange={e=>setPassword(e.target.value)}/></label>
    {error&&<div className="error">{error}</div>}<button className="primary" disabled={busy}>{busy?"Signing in…":"Sign in"}</button></form>
    <div className="note"><ShieldCheck size={17}/> Local-first AI mode is supported. Cloud Gemini is optional.</div>
  </div></div>
}

function App(){
  const [user,setUser]=useState(null),[page,setPage]=useState("dashboard"),[summary,setSummary]=useState(null),[students,setStudents]=useState([]),[selected,setSelected]=useState(null),[prediction,setPrediction]=useState(null),[shap,setShap]=useState(null),[question,setQuestion]=useState(""),[answer,setAnswer]=useState(""),[busy,setBusy]=useState(false),[err,setErr]=useState("");

  async function load(){
    setBusy(true); setErr("");
    try {
      const [s,st]=await Promise.all([api("/dashboard-summary"),api("/students")]);
      setSummary(s); setStudents(st.students||st||[]);
      const first=(st.students||st||[])[0]; if(first) await select(first.student_id||first.id);
    } catch(e){setErr(e.message)} finally{setBusy(false)}
  }
  async function select(id){
    setSelected(id); setErr("");
    try {
      const [p,x]=await Promise.all([api(`/students/${id}/prediction`),api(`/students/${id}/shap`)]);
      setPrediction(p); setShap(x);
    } catch(e){setErr(e.message)}
  }
  useEffect(()=>{ if(user) load() },[user]);
  if(!user) return <Login onLogin={setUser}/>;

  async function ask(){
    if(!question.trim()) return; setBusy(true); setAnswer("");
    try {
      const d=await api("/ai-chat",{method:"POST",body:JSON.stringify({message:question})});
      setAnswer(d.response||d.answer||d.message||JSON.stringify(d));
    } catch(e){setAnswer("AI request failed: "+e.message)} finally{setBusy(false)}
  }

  const chart=useMemo(()=>Object.entries(summary?.readiness_distribution||{}).map(([name,value])=>({name,value})),[summary]);

  return <div className="app">
    <aside><div className="brand"><Cpu/><div><b>PlacementIQ</b><span>Snapdragon AI Lab</span></div></div>
      <button className={page==="dashboard"?"nav active":"nav"} onClick={()=>setPage("dashboard")}><LayoutDashboard/>Dashboard</button>
      <button className={page==="assistant"?"nav active":"nav"} onClick={()=>setPage("assistant")}><BrainCircuit/>AI Assistant</button>
      <div className="side-bottom"><div className="device"><Cpu size={18}/><div><b>Local AI Ready</b><small>Snapdragon deployment</small></div></div><button className="nav" onClick={()=>{localStorage.removeItem("placementiq_token");setUser(null)}}><LogOut/>Logout</button></div>
    </aside>
    <main><header><div><h2>{page==="dashboard"?"Placement Dashboard":"AI Placement Assistant"}</h2><span className="muted">{user.username} · {user.role}</span></div><button className="refresh" onClick={load}><RefreshCw size={17}/> Refresh</button></header>
    {err&&<div className="error banner">{err}</div>}
    {page==="dashboard" && <Dashboard summary={summary} students={students} selected={selected} select={select} prediction={prediction} shap={shap} chart={chart}/>}
    {page==="assistant" && <div className="assistant"><div className="ai-hero"><BrainCircuit size={40}/><div><h1>Placement AI Assistant</h1><p>Ask about placement readiness, skills, projects or interview preparation.</p></div></div>
      <div className="suggestions">{["Why is my readiness score low?","What skills should I learn next?","How can I improve my placement readiness?","How should I prepare for interviews?"].map(q=><button onClick={()=>setQuestion(q)}>{q}</button>)}</div>
      <div className="chatbox"><textarea value={question} onChange={e=>setQuestion(e.target.value)} placeholder="Ask your placement question…"/><button className="primary" onClick={ask} disabled={busy}>{busy?"Thinking…":"Ask AI"}</button></div>
      {answer&&<div className="answer"><b>PlacementIQ</b><p>{answer}</p></div>}
    </div>}
    </main>
  </div>
}

function Dashboard({summary,students,selected,select,prediction,shap,chart}){
  const p=prediction?.placed_probability ?? prediction?.probability ?? prediction?.readiness_score;
  const profile=students.find(s=>(s.student_id||s.id)===selected);
  return <div>
    <div className="cards">
      <div className="card"><span>Students</span><b>{summary?.student_count??students.length}</b><small>Current dataset</small></div>
      <div className="card"><span>Predicted placement</span><b>{summary?.predicted_placement_rate!=null?summary.predicted_placement_rate+"%":"—"}</b><small>Model output</small></div>
      <div className="card"><span>Average probability</span><b>{summary?.average_placement_probability!=null?summary.average_placement_probability+"%":"—"}</b><small>Model output</small></div>
      <div className="card"><span>AI mode</span><b>Local</b><small>Snapdragon-ready architecture</small></div>
    </div>
    <div className="grid2"><section className="panel"><div className="panel-title"><h3>Student analysis</h3><select value={selected||""} onChange={e=>select(Number(e.target.value))}>{students.map(s=><option value={s.student_id||s.id}>{s.student_name||s.name||`Student ${s.student_id}`}</option>)}</select></div>
      {profile&&<div className="profile"><div className="avatar">{(profile.student_name||profile.name||"S")[0]}</div><div><b>{profile.student_name||profile.name}</b><span>{profile.branch||profile.Branch||"AI & Data Science"}</span></div></div>}
      <div className="score"><div><span>Placement readiness</span><b>{p!=null?Number(p).toFixed(1)+"%":"—"}</b></div><div className="bar"><i style={{width:`${Math.min(100,Math.max(0,Number(p)||0))}%`}}/></div>
      <div className="metrics"><div>CGPA<b>{profile?.CGPA??profile?.cgpa??"—"}</b></div><div>Technical<b>{profile?.Technical_Skill_Score??profile?.technical_skill_score??"—"}</b></div><div>Projects<b>{profile?.Projects_Count??profile?.projects_count??"—"}</b></div></div>
    </section>
    <section className="panel"><h3>Readiness distribution</h3><ResponsiveContainer width="100%" height={250}><BarChart data={chart}><XAxis dataKey="name"/><YAxis/><Tooltip/><Bar dataKey="value"/></BarChart></ResponsiveContainer></section></div>
    <section className="panel"><h3>Explainability</h3><p className="muted">SHAP explains which model inputs contributed to the selected prediction.</p><pre className="json">{JSON.stringify(shap,null,2).slice(0,5000)}</pre></section>
  </div>
}

createRoot(document.getElementById("root")).render(<App/>);
