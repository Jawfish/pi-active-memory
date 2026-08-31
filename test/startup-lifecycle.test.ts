import assert from "node:assert/strict";
import test from "node:test";
import { spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TransactionalVectorStoreError } from "../src/errors.js";
import { cleanupFailedStartup, safeStartupFailureMessage } from "../src/startup-lifecycle.js";

test("startup diagnostics preserve the VectorStore migration action without exposing arbitrary adapter errors", () => {
  const branded = new TransactionalVectorStoreError();
  branded.message = "Transactional VectorStore v2 required: secret token abc";
  assert.equal(safeStartupFailureMessage(branded), "Transactional VectorStore v2 required: migrate the RAG adapter to contractVersion: 2 with transactional mutate/compact/rebuildVectors methods");
  assert.equal(safeStartupFailureMessage(new Error("Transactional VectorStore v2 required: secret token abc")), "startup failed; check Active Memory configuration and adapters");
  assert.equal(safeStartupFailureMessage(new Error("secret token abc")), "startup failed; check Active Memory configuration and adapters");
});

test("failed startup closes and flushes partial state even when cleanup itself fails", async () => {
  const calls: string[] = [];
  await cleanupFailedStartup({
    abort: () => calls.push("abort"),
    closeStore: async () => { calls.push("close"); throw new Error("close failure"); },
    flushActivity: async () => { calls.push("flush"); throw new Error("flush failure"); },
  }, new Error("injected"));
  assert.deepEqual(calls, ["abort", "close", "flush"]);
});

test("startup repairs legacy lineage before strict embedding rebuild", async () => {
  const home = await mkdtemp(join(tmpdir(), "active-memory-lineage-order-"));
  const script = String.raw`
import { mkdir, writeFile } from 'node:fs/promises'; import { join } from 'node:path';
const cwd=join(process.env.HOME,'project'); await mkdir(join(cwd,'.pi'),{recursive:true}); await writeFile(join(cwd,'.pi','active-memory.json'),JSON.stringify({providers:{rag:{adapter:'lineage-rag',config:{storeIdentity:'lineage-store'}},embedding:{adapter:'lineage-embedding',config:{}},llm:{adapter:'lineage-llm',config:{}}},activityLog:{enabled:false,includeText:false}}));
const {default: extension}=await import('./src/index.ts'); const handlers=new Map(),commands=new Map(),events=[],notices=[];
let record={schemaVersion:1,id:'legacy',text:'legacy',kind:'fact',scope:'global',status:'active',confidence:.8,priority:.8,embeddingModel:'old-model',createdAt:'2025-01-01T00:00:00.000Z',updatedAt:'2025-01-01T00:00:00.000Z',source:{actor:'user',sessionId:'s',cwd:'/',cause:'test',reason:'test'},sourceHistory:[],supersedes:['legacy','legacy']};
const store={contractVersion:2,initialize:async()=>{},get:async()=>record,insert:async()=> 'exists',mutate:async()=>({status:'missing'}),compact:async()=>{throw Error('unused')},scan:async(_filter,visit)=>{await visit([structuredClone(record)]);return 1;},rebuildVectors:async(_dimension,build)=>{events.push('rebuild');if(record.supersedes.includes(record.id))throw Error('legacy lineage reached rebuild');const rows=await build([structuredClone(record)]);record=rows[0].record;return 1;},search:async()=>[],list:async()=>[],migrateLegacyProvenance:async()=>{events.push('migrate');record={...record,schemaVersion:2,supersedes:[]};return 1;},close:async()=>{}};
const pi={on:(name,handler)=>handlers.set(name,handler),registerMessageRenderer:()=>{},registerEntryRenderer:()=>{},registerCommand:(name,definition)=>commands.set(name,definition),registerTool:()=>{},appendEntry:()=>{},exec:async()=>({stdout:''}),sendMessage:()=>{},events:{emit:(_name,registry)=>{registry.registerRag('lineage-rag',()=>store);registry.registerEmbedding('lineage-embedding',()=>({model:'new-model',embed:async texts=>texts.map(()=>[1,0])}));registry.registerLlm('lineage-llm',()=>({json:async()=>({}),selectedModel:()=> 'lineage-model'}));}}}; extension(pi);
const ctx={cwd,mode:'rpc',hasUI:true,isProjectTrusted:()=>true,ui:{notify:message=>notices.push(String(message)),setWorkingMessage:()=>{},setStatus:()=>{},confirm:async()=>true},sessionManager:{getBranch:()=>[],getSessionFile:()=>undefined,getSessionId:()=> 'lineage',buildContextEntries:()=>[]},isIdle:()=>true,signal:new AbortController().signal};
await handlers.get('session_start')({},ctx); await commands.get('memory-status').handler('',ctx); console.log(JSON.stringify({events,status:notices.at(-1),record}));
`;
  try {
    const output = await new Promise<string>((resolve, reject) => {
      const child = spawn(process.execPath, ["--import", "tsx", "--input-type=module"], { cwd: process.cwd(), env: { ...process.env, HOME: home }, stdio: ["pipe", "pipe", "pipe"] });
      let stdout = "", stderr = "";
      child.stdout.on("data", chunk => { stdout += String(chunk); });
      child.stderr.on("data", chunk => { stderr += String(chunk); });
      child.on("error", reject);
      child.on("close", code => code === 0 ? resolve(stdout) : reject(new Error(`probe exited ${code}: ${stderr}`)));
      child.stdin.end(script);
    });
    const result = JSON.parse(output.trim()) as { events: string[]; status: string; record: { embeddingModel: string; supersedes: string[] } };
    assert.deepEqual(result.events, ["migrate", "rebuild"]);
    assert.match(result.status, /"state": "active"/);
    assert.equal(result.record.embeddingModel, "new-model");
    assert.deepEqual(result.record.supersedes, []);
  } finally { await rm(home, { recursive: true, force: true }); }
});

test("a pending next-session start hides the prior project's runtime state", async () => {
  const home = await mkdtemp(join(tmpdir(), "active-memory-startup-privacy-"));
  const script = String.raw`
import { mkdir, writeFile } from 'node:fs/promises'; import { join } from 'node:path';
const projectA=join(process.env.HOME,'project-a'), projectB=join(process.env.HOME,'project-b'); await mkdir(join(projectA,'.pi'),{recursive:true}); await mkdir(join(projectB,'.pi'),{recursive:true});
await writeFile(join(projectA,'.pi','active-memory.json'),JSON.stringify({providers:{rag:{adapter:'privacy-rag',config:{storeIdentity:'A_PRIVATE_IDENTITY'}},embedding:{adapter:'privacy-embedding',config:{}},llm:{adapter:'privacy-llm',config:{}}},prompts:{tools:{memorySearch:{snippet:'A_PRIVATE_PROMPT'}}},activityLog:{enabled:false,includeText:false}}));
await writeFile(join(projectB,'.pi','active-memory.json'),'[]');
const {default: extension}=await import('./src/index.ts');
const handlers=new Map(), commands=new Map(), tools=new Map(), notices=[]; let releaseClose, closeStartedResolve; const closeGate=new Promise(resolve=>releaseClose=resolve), closeStarted=new Promise(resolve=>closeStartedResolve=resolve);
const store={contractVersion:2,initialize:async()=>{},get:async()=>undefined,insert:async()=> 'inserted',mutate:async()=>({status:'missing'}),compact:async()=>{throw Error('unused')},scan:async()=>0,rebuildVectors:async()=>0,search:async()=>[],list:async()=>[],migrateLegacyProvenance:async()=>0,close:async()=>{closeStartedResolve();await closeGate;}};
const pi={on:(name,handler)=>handlers.set(name,handler),registerMessageRenderer:()=>{},registerEntryRenderer:()=>{},registerCommand:(name,definition)=>commands.set(name,definition),registerTool:definition=>tools.set(definition.name,definition),appendEntry:()=>{},exec:async()=>({stdout:''}),sendMessage:()=>{},events:{emit:(_name,registry)=>{registry.registerRag('privacy-rag',()=>store);registry.registerEmbedding('privacy-embedding',()=>({model:'privacy-model',embed:async texts=>texts.map(()=>[1,0])}));registry.registerLlm('privacy-llm',()=>({json:async()=>({}),selectedModel:()=> 'privacy-model'}));}}}; extension(pi);
function context(cwd,id){return {cwd,mode:'rpc',hasUI:false,isProjectTrusted:()=>true,ui:{notify:message=>notices.push(String(message)),setWorkingMessage:()=>{},setStatus:()=>{},confirm:async()=>true},sessionManager:{getBranch:()=>[],getSessionFile:()=>undefined,getSessionId:()=>id,buildContextEntries:()=>[]},isIdle:()=>true,signal:new AbortController().signal};}
const start=handlers.get('session_start'); await start({},context(projectA,'A')); if(tools.get('memory_search').promptSnippet!=='A_PRIVATE_PROMPT')throw Error('project A did not publish');
const pending=start({},context(projectB,'B')); await closeStarted;
const promptDuring=tools.get('memory_search').promptSnippet; await commands.get('memory-status').handler('',context(projectB,'status')); const statusDuring=notices.at(-1); let searchError=''; try{await tools.get('memory_search').execute('id',{query:'secret'},new AbortController().signal,()=>{},context(projectB,'tool'));}catch(error){searchError=String(error.message);}
releaseClose(); await pending; const promptAfter=tools.get('memory_search').promptSnippet; console.log(JSON.stringify({promptDuring,statusDuring,searchError,promptAfter}));
`;
  try {
    const output = await new Promise<string>((resolve, reject) => {
      const child = spawn(process.execPath, ["--import", "tsx", "--input-type=module"], { cwd: process.cwd(), env: { ...process.env, HOME: home }, stdio: ["pipe", "pipe", "pipe"] });
      let stdout = "", stderr = "";
      child.stdout.on("data", chunk => { stdout += String(chunk); });
      child.stderr.on("data", chunk => { stderr += String(chunk); });
      child.on("error", reject);
      child.on("close", code => code === 0 ? resolve(stdout) : reject(new Error(`probe exited ${code}: ${stderr}`)));
      child.stdin.end(script);
    });
    const result = JSON.parse(output.trim()) as { promptDuring: string; statusDuring: string; searchError: string; promptAfter: string };
    assert.doesNotMatch(JSON.stringify(result), /A_PRIVATE/);
    assert.match(result.statusDuring, /"projectId": "uninitialized"/);
    assert.match(result.searchError, /not initialized/);
    assert.equal(result.promptAfter, result.promptDuring);
  } finally { await rm(home, { recursive: true, force: true }); }
});

test("in-flight tools and fast-model callbacks remain bound to their session", async () => {
  const home = await mkdtemp(join(tmpdir(), "active-memory-generation-tools-"));
  const script = String.raw`
import { mkdir, writeFile } from 'node:fs/promises'; import { join } from 'node:path';
const projectA=join(process.env.HOME,'project-a'),projectB=join(process.env.HOME,'project-b');for(const cwd of [projectA,projectB]){await mkdir(join(cwd,'.pi'),{recursive:true});await writeFile(join(cwd,'.pi','active-memory.json'),JSON.stringify({providers:{rag:{adapter:'cross-rag',config:{storeIdentity:cwd}},embedding:{adapter:'cross-embedding',config:{}},llm:{adapter:'cross-llm',config:{}}},activityLog:{enabled:false,includeText:false}}));}
const {default:extension}=await import('./src/index.ts');const handlers=new Map(),commands=new Map(),tools=new Map(),tokenCallbacks=new Map(),branch=[],notices=[];let releaseEmbed,embedStartedResolve;const embedGate=new Promise(resolve=>releaseEmbed=resolve),embedStarted=new Promise(resolve=>embedStartedResolve=resolve);
const memory=id=>({schemaVersion:2,id,text:id+'_PRIVATE_MEMORY',kind:'fact',scope:'global',status:'active',confidence:.8,priority:.8,embeddingModel:'model',createdAt:'2025-01-01T00:00:00.000Z',updatedAt:'2025-01-01T00:00:00.000Z',source:{actor:'user',sessionId:id,cwd:'/',cause:'test',reason:'test'},sourceHistory:[]});
function store(id){return{contractVersion:2,initialize:async()=>{},get:async()=>undefined,insert:async()=> 'exists',mutate:async()=>({status:'missing'}),compact:async()=>{throw Error('unused')},scan:async()=>0,rebuildVectors:async()=>0,search:async()=>[{record:memory(id),score:1}],list:async()=>[],migrateLegacyProvenance:async()=>0,close:async()=>{}}}
const pi={on:(name,handler)=>handlers.set(name,handler),registerMessageRenderer:()=>{},registerEntryRenderer:()=>{},registerCommand:(name,definition)=>commands.set(name,definition),registerTool:definition=>tools.set(definition.name,definition),appendEntry:(type,data)=>branch.push({type:'custom',customType:type,data}),exec:async()=>({stdout:''}),sendMessage:()=>{},events:{emit:(_name,registry)=>{registry.registerRag('cross-rag',(_config,context)=>store(context.sessionId));registry.registerEmbedding('cross-embedding',(_config,context)=>({model:'model',embed:async texts=>{if(context.sessionId==='A'){embedStartedResolve();await embedGate;throw Error('A_PRIVATE_ADAPTER_ERROR');}return texts.map(()=>[1,0]);}}));registry.registerLlm('cross-llm',(_config,context)=>({json:async()=>({}),selectedModel:()=> 'model',onTokenUsage:callback=>tokenCallbacks.set(context.sessionId,callback)}));}}};extension(pi);
function context(cwd,id){return{cwd,mode:'rpc',hasUI:false,isProjectTrusted:()=>true,ui:{notify:message=>notices.push(String(message)),setWorkingMessage:()=>{},setStatus:()=>{},confirm:async()=>true},sessionManager:{getBranch:()=>branch,getSessionFile:()=>undefined,getSessionId:()=>id,buildContextEntries:()=>[]},isIdle:()=>true,signal:new AbortController().signal};}
const start=handlers.get('session_start');await start({},context(projectA,'A'));const search=tools.get('memory_search').execute('search',{query:'q'},new AbortController().signal,()=>{},context(projectA,'A')).then(value=>({value}),error=>({error:String(error.message)}));await embedStarted;const feedback=tools.get('memory_feedback').execute('feedback',{steerToken:'unknown',memoryId:'A_PRIVATE_ID',outcome:'useful',reason:'test'}).then(value=>({value}),error=>({error:String(error.message)}));const next=start({},context(projectB,'B'));tokenCallbacks.get('A')({input:41,output:7});releaseEmbed();const searchResult=await search,feedbackResult=await feedback;await next;await commands.get('memory-stats').handler('',context(projectB,'B'));console.log(JSON.stringify({searchResult,feedbackResult,stats:notices.at(-1),branch}));
`;
  try {
    const output = await new Promise<string>((resolve, reject) => {
      const child = spawn(process.execPath, ["--import", "tsx", "--input-type=module"], { cwd: process.cwd(), env: { ...process.env, HOME: home }, stdio: ["pipe", "pipe", "pipe"] });
      let stdout = "", stderr = "";
      child.stdout.on("data", chunk => { stdout += String(chunk); });
      child.stderr.on("data", chunk => { stderr += String(chunk); });
      child.on("error", reject);
      child.on("close", code => code === 0 ? resolve(stdout) : reject(new Error(`probe exited ${code}: ${stderr}`)));
      child.stdin.end(script);
    });
    const result = JSON.parse(output.trim()) as { searchResult: { error?: string; value?: unknown }; feedbackResult: { error?: string; value?: unknown }; stats: string; branch: unknown[] };
    assert.match(result.searchResult.error ?? "", /session changed/);
    assert.match(result.feedbackResult.error ?? "", /session changed/);
    assert.doesNotMatch(JSON.stringify([result.searchResult, result.feedbackResult]), /B_PRIVATE_MEMORY|A_PRIVATE_ID/);
    assert.doesNotMatch(result.stats, /41|7/);
    assert.deepEqual(result.branch, []);
  } finally { await rm(home, { recursive: true, force: true }); }
});

test("delayed commands cannot read A then mutate B", async () => {
  const home = await mkdtemp(join(tmpdir(), "active-memory-command-generation-"));
  const script = String.raw`
import {mkdir,writeFile} from 'node:fs/promises';import{join}from'node:path';const a=join(process.env.HOME,'a'),b=join(process.env.HOME,'b');for(const cwd of[a,b]){await mkdir(join(cwd,'.pi'),{recursive:true});await writeFile(join(cwd,'.pi','active-memory.json'),JSON.stringify({providers:{rag:{adapter:'command-rag',config:{storeIdentity:cwd}},embedding:{adapter:'command-embedding',config:{}},llm:{adapter:'command-llm',config:{}}},activityLog:{enabled:false,includeText:false}}));}
const{default:extension}=await import('./src/index.ts');const handlers=new Map(),commands=new Map(),events=[],notices=[];let block=false,releaseList,listStartedResolve;const listGate=new Promise(resolve=>releaseList=resolve),listStarted=new Promise(resolve=>listStartedResolve=resolve);const record=id=>({schemaVersion:2,id:'same',text:id+'_PRIVATE',kind:'fact',scope:'global',status:'active',confidence:.8,priority:.8,embeddingModel:'model',createdAt:'2025-01-01T00:00:00.000Z',updatedAt:'2025-01-01T00:00:00.000Z',source:{actor:'user',sessionId:id,cwd:'/',cause:'test',reason:'test'},sourceHistory:[]});function store(id){return{contractVersion:2,initialize:async()=>{},get:async()=>undefined,insert:async()=> 'exists',mutate:async()=>{events.push('mutate:'+id);return{status:'updated',record:record(id)}},compact:async()=>{throw Error('unused')},scan:async()=>0,rebuildVectors:async()=>0,search:async()=>[],list:async()=>{if(id==='A'&&block){events.push('list:A:start');listStartedResolve();await listGate;events.push('list:A:end');return[record('A')];}return[]},migrateLegacyProvenance:async()=>0,close:async()=>events.push('close:'+id)}}
const pi={on:(n,h)=>handlers.set(n,h),registerMessageRenderer:()=>{},registerEntryRenderer:()=>{},registerCommand:(n,d)=>commands.set(n,d),registerTool:()=>{},appendEntry:()=>{},exec:async()=>({stdout:''}),sendMessage:()=>{},events:{emit:(_n,r)=>{r.registerRag('command-rag',(_c,x)=>store(x.sessionId));r.registerEmbedding('command-embedding',()=>({model:'model',embed:async texts=>texts.map(()=>[1,0])}));r.registerLlm('command-llm',()=>({json:async()=>({}),selectedModel:()=> 'model'}));}}};extension(pi);function ctx(cwd,id){return{cwd,mode:'rpc',hasUI:false,isProjectTrusted:()=>true,ui:{notify:m=>notices.push(String(m)),setWorkingMessage:()=>{},setStatus:()=>{},confirm:async()=>{events.push('confirm');return true}},sessionManager:{getBranch:()=>[],getSessionFile:()=>undefined,getSessionId:()=>id,buildContextEntries:()=>[]},isIdle:()=>true,signal:new AbortController().signal};}
const start=handlers.get('session_start');await start({},ctx(a,'A'));block=true;const forgetting=commands.get('memory-forget').handler('same',ctx(a,'A')).then(value=>({value}),error=>({error:String(error.message)}));await listStarted;const next=start({},ctx(b,'B'));releaseList();const forgetResult=await forgetting;await next;console.log(JSON.stringify({events,notices,forgetResult}));
`;
  try {
    const output = await new Promise<string>((resolve, reject) => {
      const child = spawn(process.execPath, ["--import", "tsx", "--input-type=module"], { cwd: process.cwd(), env: { ...process.env, HOME: home }, stdio: ["pipe", "pipe", "pipe"] });
      let stdout = "", stderr = "";
      child.stdout.on("data", chunk => { stdout += String(chunk); }); child.stderr.on("data", chunk => { stderr += String(chunk); }); child.on("error", reject);
      child.on("close", code => code === 0 ? resolve(stdout) : reject(new Error(`probe exited ${code}: ${stderr}`))); child.stdin.end(script);
    });
    const result = JSON.parse(output.trim()) as { events: string[]; notices: string[]; forgetResult: { error?: string } };
    assert.match(result.forgetResult.error ?? "", /session changed/);
    assert.deepEqual(result.events, ["list:A:start", "list:A:end", "close:A"]);
    assert.doesNotMatch(result.notices.join("\n"), /A_PRIVATE|B_PRIVATE/);
  } finally { await rm(home, { recursive: true, force: true }); }
});

test("overlapping session starts close only the stale generation's local store", async () => {
  const home = await mkdtemp(join(tmpdir(), "active-memory-startup-race-"));
  const script = String.raw`
import { mkdir, writeFile } from 'node:fs/promises'; import { join } from 'node:path';
const cwd=join(process.env.HOME,'project'); await mkdir(join(cwd,'.pi'),{recursive:true});
await writeFile(join(cwd,'.pi','active-memory.json'),JSON.stringify({providers:{rag:{adapter:'race-rag',config:{storeIdentity:'race-store'}},embedding:{adapter:'race-embedding',config:{}},llm:{adapter:'race-llm',config:{}}},activityLog:{enabled:false,includeText:false}}));
const {default: extension}=await import('./src/index.ts');
const handlers=new Map(), commands=new Map(), events=[]; let releaseA; const gate=new Promise(resolve=>releaseA=resolve); let startedA; const started=new Promise(resolve=>startedA=resolve);
function makeStore(id){return {contractVersion:2,initialize:async()=>{events.push('init:'+id);if(id==='A'){startedA();await gate;}},get:async()=>undefined,insert:async()=> 'inserted',mutate:async()=>({status:'missing'}),compact:async()=>{throw Error('unused')},scan:async()=>0,rebuildVectors:async()=>0,search:async()=>[],list:async()=>[],migrateLegacyProvenance:async()=>0,close:async()=>events.push('close:'+id)}}
const pi={on:(name,handler)=>handlers.set(name,handler),registerMessageRenderer:()=>{},registerEntryRenderer:()=>{},registerCommand:(name,definition)=>commands.set(name,definition),registerTool:()=>{},appendEntry:()=>{},exec:async()=>({stdout:''}),sendMessage:()=>{},events:{emit:(_name,registry)=>{registry.registerRag('race-rag',(_config,context)=>makeStore(context.sessionId));registry.registerEmbedding('race-embedding',()=>({model:'race-model',embed:async texts=>texts.map(()=>[1,0])}));registry.registerLlm('race-llm',()=>({json:async()=>({}),selectedModel:()=> 'race-model'}));}}}; extension(pi);
const notices=[]; function context(id){return {cwd,mode:'rpc',hasUI:false,isProjectTrusted:()=>true,ui:{notify:message=>notices.push(String(message)),setWorkingMessage:()=>{},setStatus:()=>{},confirm:async()=>true},sessionManager:{getBranch:()=>[],getSessionFile:()=>undefined,getSessionId:()=>id,buildContextEntries:()=>[]},isIdle:()=>true,signal:new AbortController().signal};}
const start=handlers.get('session_start'); const first=start({},context('A')); await started; const second=start({},context('B')); releaseA(); await Promise.all([first,second]); await commands.get('memory-status').handler('',context('status')); console.log(JSON.stringify({events,status:notices.at(-1)}));
`;
  try {
    const output = await new Promise<string>((resolve, reject) => {
      const child = spawn(process.execPath, ["--import", "tsx", "--input-type=module"], { cwd: process.cwd(), env: { ...process.env, HOME: home }, stdio: ["pipe", "pipe", "pipe"] });
      let stdout = "", stderr = "";
      child.stdout.on("data", chunk => { stdout += String(chunk); });
      child.stderr.on("data", chunk => { stderr += String(chunk); });
      child.on("error", reject);
      child.on("close", code => code === 0 ? resolve(stdout) : reject(new Error(`probe exited ${code}: ${stderr}`)));
      child.stdin.end(script);
    });
    const result = JSON.parse(output.trim()) as { events: string[]; status: string };
    assert.deepEqual(result.events, ["init:A", "close:A", "init:B"]);
    assert.match(result.status, /"state": "active"/);
  } finally { await rm(home, { recursive: true, force: true }); }
});
