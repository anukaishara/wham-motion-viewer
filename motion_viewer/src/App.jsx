import React, { Suspense, useCallback, useEffect, useState, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, useGLTF, Environment, ContactShadows } from '@react-three/drei';
import * as THREE from 'three';

const CONTACT_THRESHOLD = 0.5;

const SMPL_BONE_ORDER = [
  'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
  'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar',
  'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
  'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
];

const WHAM_STAGES = [
  { id: 1, label: 'Preprocess video' },
  { id: 2, label: 'Detect & track' },
  { id: 2, label: 'WHAM inference' },
  { id: 3, label: 'Extract parameters' },
];

function applySmoothing(data, windowSize = 2) {
  if (!data || data.length === 0) return [];
  const smoothed = [];
  for (let i = 0; i < data.length; i++) {
    const start = Math.max(0, i - windowSize);
    const end   = Math.min(data.length - 1, i + windowSize);
    const count = end - start + 1;
    const row   = new Array(data[0].length).fill(0);
    for (let j = start; j <= end; j++)
      for (let k = 0; k < row.length; k++) row[k] += data[j][k];
    for (let k = 0; k < row.length; k++) row[k] /= count;
    smoothed.push(row);
  }
  return smoothed;
}

// ============================================================================
// Upload Screen
// ============================================================================
function UploadScreen({ onUpload }) {
  const [dragging,  setDragging]  = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error,     setError]     = useState(null);
  const fileRef = useRef();

  const submit = async (file) => {
    if (!file) return;
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['mp4', 'mov', 'avi', 'mkv', 'webm'].includes(ext)) {
      setError(`Unsupported format: .${ext}. Use .mp4, .mov, .avi, .mkv or .webm`);
      return;
    }
    setUploading(true);
    setError(null);
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`Server error ${res.status}: ${await res.text()}`);
      const { job_id } = await res.json();
      onUpload(job_id);
    } catch (e) {
      setError(e.message);
      setUploading(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) submit(file);
  };

  return (
    <div style={s.page}>
      <div style={s.card}>
        <div style={s.logo}>
          <span style={s.logoW}>W</span>
          <span style={s.logoH}>H</span>
          <span style={s.logoA}>A</span>
          <span style={s.logoM}>M</span>
        </div>
        <h1 style={s.title}>3D Motion Capture</h1>
        <p style={s.subtitle}>
          Upload a video to reconstruct world-space 3D human motion
        </p>

        <div
          style={{ ...s.dropzone, ...(dragging ? s.dropzoneActive : {}) }}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => !uploading && fileRef.current?.click()}
        >
          {uploading ? (
            <>
              <div style={s.spinner} />
              <p style={s.dropText}>Uploading…</p>
            </>
          ) : (
            <>
              <svg style={{ marginBottom: 12 }} width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="16 16 12 12 8 16" />
                <line x1="12" y1="12" x2="12" y2="21" />
                <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" />
              </svg>
              <p style={s.dropText}>
                {dragging ? 'Drop to upload' : 'Drag & drop a video, or click to browse'}
              </p>
              <p style={s.dropHint}>MP4 · MOV · AVI · MKV · WEBM</p>
            </>
          )}
        </div>

        <input
          ref={fileRef}
          type="file"
          accept="video/*"
          style={{ display: 'none' }}
          onChange={(e) => submit(e.target.files[0])}
        />

        {error && (
          <div style={s.errorBox}>⚠ {error}</div>
        )}

        <div style={s.infoRow}>
          <span style={s.infoItem}>🎮 RTX 5090</span>
          <span style={s.infoItem}>⚡ ~60 s per minute of video</span>
          <span style={s.infoItem}>📐 SMPL body model</span>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Processing / Progress Screen
// ============================================================================
function ProcessingScreen({ jobId, onDone, onError }) {
  const [prog, setProg] = useState({
    stage: 0, stage_name: 'Starting pipeline…', pct: 1, status: 'running',
  });
  const [log, setLog]     = useState([]);
  const [etaStr, setEta]  = useState('');

  useEffect(() => {
    const es = new EventSource(`/api/progress/${jobId}`);

    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        setProg(data);

        // Build ETA string
        if (data.eta_s != null) {
          const s = Math.round(data.eta_s);
          if (s < 60)  setEta(`~${s}s remaining`);
          else         setEta(`~${Math.round(s / 60)}m remaining`);
        } else {
          setEta('');
        }

        // Rolling log (last 6 messages)
        if (data.stage_name) {
          setLog(prev => {
            const msg = data.stage_name;
            if (prev[prev.length - 1] === msg) return prev;
            return [...prev.slice(-5), msg];
          });
        }

        if (data.status === 'done')  { es.close(); onDone(); }
        if (data.status === 'error') { es.close(); onError(data.message || 'Pipeline failed'); }
      } catch {}
    };

    es.onerror = () => {
      es.close();
      onError('Lost connection to server');
    };

    return () => es.close();
  }, [jobId]);

  const pct = Math.min(100, Math.max(0, prog.pct ?? 0));

  // Stage indicator: map pct → active step index
  const stepIdx = pct < 25 ? 0 : pct < 55 ? 1 : pct < 80 ? 2 : 3;

  return (
    <div style={s.page}>
      <div style={{ ...s.card, maxWidth: 520 }}>
        <div style={s.logo}>
          <span style={s.logoW}>W</span>
          <span style={s.logoH}>H</span>
          <span style={s.logoA}>A</span>
          <span style={s.logoM}>M</span>
        </div>
        <h2 style={{ ...s.title, fontSize: 22, marginBottom: 6 }}>Processing</h2>
        <p style={{ ...s.subtitle, marginBottom: 24 }}>Job {jobId}</p>

        {/* Step indicators */}
        <div style={{ display: 'flex', gap: 6, marginBottom: 28 }}>
          {WHAM_STAGES.map((st, i) => {
            const done   = i < stepIdx;
            const active = i === stepIdx;
            return (
              <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
                <div style={{
                  width: 32, height: 32, borderRadius: '50%',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 13, fontWeight: 700,
                  background: done   ? 'linear-gradient(135deg,#10b981,#059669)'
                             : active ? 'linear-gradient(135deg,#3b82f6,#8b5cf6)'
                             : '#1e293b',
                  border: active ? '2px solid #60a5fa' : '2px solid transparent',
                  color: (done || active) ? 'white' : '#475569',
                  transition: 'all 0.3s',
                }}>
                  {done ? '✓' : i + 1}
                </div>
                <span style={{ fontSize: 9, color: active ? '#93c5fd' : done ? '#6ee7b7' : '#475569', textAlign: 'center', lineHeight: 1.3, fontWeight: active ? 700 : 400 }}>
                  {st.label}
                </span>
              </div>
            );
          })}
        </div>

        {/* Progress bar */}
        <div style={{ background: '#1e293b', borderRadius: 8, height: 10, overflow: 'hidden', marginBottom: 10 }}>
          <div style={{
            height: '100%', borderRadius: 8,
            background: 'linear-gradient(90deg,#3b82f6,#8b5cf6)',
            width: `${pct}%`,
            transition: 'width 0.4s ease',
            boxShadow: '0 0 10px rgba(99,102,241,0.5)',
          }} />
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 20 }}>
          <span style={{ fontSize: 12, color: '#94a3b8' }}>{prog.stage_name}</span>
          <span style={{ fontSize: 12, color: '#60a5fa', fontWeight: 700 }}>{pct}%</span>
        </div>

        {etaStr && (
          <p style={{ fontSize: 12, color: '#64748b', textAlign: 'center', marginBottom: 16 }}>{etaStr}</p>
        )}

        {/* Rolling log */}
        <div style={{ background: 'rgba(0,0,0,0.3)', borderRadius: 8, padding: '10px 14px', minHeight: 72 }}>
          {log.map((msg, i) => (
            <div key={i} style={{
              fontSize: 11, fontFamily: 'monospace',
              color: i === log.length - 1 ? '#e2e8f0' : '#475569',
              marginBottom: 2,
            }}>
              {i === log.length - 1 ? '▶ ' : '  '}{msg}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Ground Contact Markers
// ============================================================================
function ContactMarkers({ contactFrame, leftFootRef, rightFootRef, groundY }) {
  const lRingRef = useRef();
  const rRingRef = useRef();
  const tmpVec   = new THREE.Vector3();

  useFrame(() => {
    if (!contactFrame) return;
    const leftContact  = (contactFrame[0] + contactFrame[2]) / 2 > CONTACT_THRESHOLD;
    const rightContact = (contactFrame[1] + contactFrame[3]) / 2 > CONTACT_THRESHOLD;

    if (lRingRef.current) {
      lRingRef.current.visible = leftContact;
      if (leftContact && leftFootRef.current) {
        leftFootRef.current.getWorldPosition(tmpVec);
        lRingRef.current.position.set(tmpVec.x, groundY, tmpVec.z);
      }
    }
    if (rRingRef.current) {
      rRingRef.current.visible = rightContact;
      if (rightContact && rightFootRef.current) {
        rightFootRef.current.getWorldPosition(tmpVec);
        rRingRef.current.position.set(tmpVec.x, groundY, tmpVec.z);
      }
    }
  });

  const RingMesh = ({ meshRef, color }) => (
    <mesh ref={meshRef} rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[0.06, 0.12, 32]} />
      <meshBasicMaterial color={color} transparent opacity={0.85} side={THREE.DoubleSide} />
    </mesh>
  );

  return (
    <>
      <RingMesh meshRef={lRingRef} color="#38bdf8" />
      <RingMesh meshRef={rRingRef} color="#a78bfa" />
    </>
  );
}

// ============================================================================
// Camera Follow Rig
// ============================================================================
function CameraFollow({ avatarGroupRef, controlsRef }) {
  const smoothTarget = useRef(new THREE.Vector3());
  const initialized  = useRef(false);

  useFrame(() => {
    if (!avatarGroupRef.current || !controlsRef.current) return;
    const p = avatarGroupRef.current.position;
    if (!initialized.current) {
      smoothTarget.current.set(p.x, p.y + 0.9, p.z);
      initialized.current = true;
    } else {
      smoothTarget.current.lerp(new THREE.Vector3(p.x, p.y + 0.9, p.z), 0.12);
    }
    controlsRef.current.target.copy(smoothTarget.current);
  });
  return null;
}

// ============================================================================
// Core Kinematic Animator
// ============================================================================
function SMPLAnimator({ gender, motionData, currentFrame, betaData, transData, contactData, onGroundY, groupRef }) {
  const modelPath = `/models/smpl_${gender}.glb`;
  const { scene }  = useGLTF(modelPath);

  const boneMap      = useRef({});
  const scaleRef     = useRef(1);
  const initDone     = useRef(false);
  const leftFootRef  = useRef();
  const rightFootRef = useRef();
  const groundYRef   = useRef(-1.05);

  useEffect(() => {
    if (!scene) return;
    initDone.current = false;
    boneMap.current  = {};

    scene.updateMatrixWorld(true);
    const box    = new THREE.Box3().setFromObject(scene);
    const size   = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    scaleRef.current = maxDim > 0 && isFinite(maxDim) ? 2.0 / maxDim : 1;

    groundYRef.current = box.min.y * scaleRef.current;
    if (onGroundY) onGroundY(groundYRef.current);

    const map = {};
    scene.traverse((child) => {
      if (child.isBone) {
        const clean = child.name.replace(/^[mf]_avg_/, '');
        map[clean] = child;
      }
      if (child.isSkinnedMesh) {
        child.frustumCulled = false;
        child.castShadow    = true;
        child.receiveShadow = true;
      }
    });
    boneMap.current = map;

    leftFootRef.current  = map['L_Foot'] || null;
    rightFootRef.current = map['R_Foot'] || null;
    initDone.current = true;
  }, [scene]);

  useEffect(() => {
    if (!scene || !betaData || betaData.length < 1) return;
    const BETA_SCALE = 0.15;
    scene.traverse((child) => {
      if (!child.isSkinnedMesh || !child.morphTargetDictionary) return;
      if (child.morphTargetInfluences) child.morphTargetInfluences.fill(0);
      for (let i = 0; i < Math.min(10, betaData.length); i++) {
        const label = String(i).padStart(3, '0');
        const beta  = betaData[i] * BETA_SCALE;
        if (beta > 0) {
          const idx = child.morphTargetDictionary[`Shape${label}_pos`];
          if (idx !== undefined) child.morphTargetInfluences[idx] = Math.min(beta, 1.0);
        } else if (beta < 0) {
          const idx = child.morphTargetDictionary[`Shape${label}_neg`];
          if (idx !== undefined) child.morphTargetInfluences[idx] = Math.min(-beta, 1.0);
        }
      }
    });
  }, [scene, betaData]);

  useFrame(() => {
    if (!initDone.current || !motionData || motionData.length === 0) return;
    const frameData = motionData[currentFrame];
    if (!frameData) return;

    if (transData && transData[currentFrame] && groupRef.current) {
      const td  = transData[currentFrame];
      const off = transData[0];
      const s   = scaleRef.current;
      groupRef.current.position.set(
        (td[0] - off[0]) * s,
        (td[1] - off[1]) * s,
        (td[2] - off[2]) * s
      );
    }

    const q    = new THREE.Quaternion();
    const axis = new THREE.Vector3();
    for (let i = 0; i < 24; i++) {
      const rx    = frameData[i * 3];
      const ry    = frameData[i * 3 + 1];
      const rz    = frameData[i * 3 + 2];
      const angle = Math.sqrt(rx * rx + ry * ry + rz * rz);
      const bone  = boneMap.current[SMPL_BONE_ORDER[i]];
      if (!bone) continue;
      if (angle > 1e-4) {
        axis.set(rx / angle, ry / angle, rz / angle);
        q.setFromAxisAngle(axis, angle);
      } else {
        q.identity();
      }
      bone.quaternion.copy(q);
    }
  });

  const contactFrame = contactData && contactData[currentFrame];

  return (
    <>
      <group ref={groupRef}>
        <primitive object={scene} scale={scaleRef.current} />
      </group>
      <ContactMarkers
        contactFrame={contactFrame}
        leftFootRef={leftFootRef}
        rightFootRef={rightFootRef}
        groundY={groundYRef.current}
      />
    </>
  );
}

// ============================================================================
// Canvas Recorder
// ============================================================================
function CanvasRecorder({ isRecording, onFrameCaptured, onStop }) {
  const { gl }       = useThree();
  const recorderRef  = useRef(null);
  const chunksRef    = useRef([]);
  const activeRef    = useRef(false);

  useEffect(() => {
    if (isRecording && !activeRef.current) {
      chunksRef.current = [];
      const stream   = gl.domElement.captureStream(30);
      const mimeType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
        .find(m => MediaRecorder.isTypeSupported(m)) || '';
      const recorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType, videoBitsPerSecond: 8_000_000 } : undefined
      );
      recorder.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'video/webm' });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = `smpl_animation_${Date.now()}.webm`;
        a.click();
        URL.revokeObjectURL(url);
        onStop();
      };
      recorder.start();
      recorderRef.current = recorder;
      activeRef.current   = true;
    } else if (!isRecording && activeRef.current) {
      recorderRef.current?.stop();
      activeRef.current = false;
    }
  }, [isRecording]);

  useFrame(() => { if (activeRef.current) onFrameCaptured(); });
  return null;
}

// ============================================================================
// Viewer Application (the 3D canvas + sidebar)
// ============================================================================
function ViewerApp({ dataBaseUrl, onNewVideo }) {
  const [motionData,        setMotionData]        = useState([]);
  const [betaData,          setBetaData]          = useState([]);
  const [transData,         setTransData]         = useState([]);
  const [contactData,       setContactData]       = useState([]);
  const [currentFrame,      setCurrentFrame]      = useState(0);
  const [isPlaying,         setIsPlaying]         = useState(false);
  const [gender,            setGender]            = useState('female');
  const [dataReady,         setDataReady]         = useState(false);
  const [groundY,           setGroundY]           = useState(-1.05);
  const [cameraMode,        setCameraMode]        = useState('follow');
  const [isRecording,       setIsRecording]       = useState(false);
  const [recProgress,       setRecProgress]       = useState(0);
  const [metadata,          setMetadata]          = useState(null);
  const [viewMode,          setViewMode]          = useState('3d');
  const [videoUrl,          setVideoUrl]          = useState(null);
  const [videoOffsetFrames, setVideoOffsetFrames] = useState(0);
  const [loadWarnings,      setLoadWarnings]      = useState([]);

  const controlsRef    = useRef();
  const avatarGroupRef = useRef();
  const recFrameRef    = useRef(0);
  const recStartRef    = useRef(0);
  const videoRef       = useRef(null);

  const playbackFps = metadata?.export_fps ?? metadata?.processed_fps ?? metadata?.source_fps ?? 30;
  const totalFrames = Math.max(0, motionData.length - 1);
  const inSplitMode = viewMode === 'split' && !!videoUrl;

  // ── Load data from dataBaseUrl ──────────────────────────────────────────
  useEffect(() => {
    setMotionData([]); setBetaData([]); setTransData([]); setContactData([]);
    setDataReady(false); setMetadata(null);
    setVideoUrl(null); setLoadWarnings([]); setCurrentFrame(0); setIsPlaying(false);

    const base = dataBaseUrl;

    Promise.all([
      fetch(`${base}/thetas.csv`).then(r => r.text()),
      fetch(`${base}/betas.json`).then(r => r.json()),
      fetch(`${base}/trans.csv`).then(r => r.text()),
    ]).then(([thetaText, betaJson, transText]) => {
      setMotionData(applySmoothing(thetaText.trim().split('\n').map(row => row.split(',').map(Number)), 2));
      if (betaJson?.betas) setBetaData(betaJson.betas);
      setTransData(applySmoothing(transText.trim().split('\n').map(row => row.split(',').map(Number)), 2));
      setDataReady(true);
      setIsPlaying(true);
    }).catch(err => console.error('Core data load error:', err));

    fetch(`${base}/contact.csv`)
      .then(r => r.text())
      .then(t => setContactData(t.trim().split('\n').map(row => row.split(',').map(Number))))
      .catch(() => {});

    fetch(`${base}/metadata.json`)
      .then(r => { if (!r.ok) throw new Error('not found'); return r.json(); })
      .then(meta => {
        setMetadata(meta);
        if (meta.warnings?.length > 0)
          setLoadWarnings(prev => [...prev, ...meta.warnings]);
      })
      .catch(() => setLoadWarnings(prev => {
        const msg = 'metadata.json not found — using 30 fps';
        return prev.includes(msg) ? prev : [...prev, msg];
      }));

    (async () => {
      for (const path of [`${base}/processed_video.mp4`, `${base}/input_video.mp4`]) {
        try {
          const r = await fetch(path);
          if (r.ok) {
            r.body?.cancel().catch(() => {});
            setVideoUrl(path);
            return;
          }
        } catch {}
      }
    })();
  }, [dataBaseUrl]);

  // ── Playback timer ─────────────────────────────────────────────────────
  useEffect(() => {
    if (!isPlaying || motionData.length === 0) return;
    const timer = setInterval(
      () => setCurrentFrame(prev => (prev + 1) % motionData.length),
      1000 / playbackFps
    );
    return () => clearInterval(timer);
  }, [isPlaying, motionData.length, playbackFps]);

  // ── Video sync ─────────────────────────────────────────────────────────
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !videoUrl) return;
    const vf = Math.max(0, Math.min(currentFrame + videoOffsetFrames, totalFrames));
    video.currentTime = vf / playbackFps;
  }, [currentFrame, videoOffsetFrames, playbackFps, videoUrl, totalFrames, viewMode]);

  // ── Recording ──────────────────────────────────────────────────────────
  const startRecording = useCallback(() => {
    if (!dataReady || isRecording) return;
    setCurrentFrame(0); setIsPlaying(false);
    recFrameRef.current = 0; recStartRef.current = 0;
    setRecProgress(0); setIsRecording(true);
  }, [dataReady, isRecording]);

  const handleRecordFrame = useCallback(() => {
    recFrameRef.current += 1;
    setRecProgress(Math.min(100, Math.round((recFrameRef.current / (totalFrames + 1)) * 100)));
    if (recFrameRef.current > totalFrames) setIsRecording(false);
    else setCurrentFrame(recFrameRef.current);
  }, [totalFrames]);

  const handleRecordStop = useCallback(() => { setIsRecording(false); setRecProgress(0); }, []);

  // ── Sync status ────────────────────────────────────────────────────────
  const thetaCount    = motionData.length;
  const videoCount    = metadata?.source_frame_count ?? metadata?.processed_frame_count ?? null;
  const metaTheta     = metadata?.theta_frame_count ?? null;
  const frameMismatch = videoCount !== null && thetaCount > 0
    && Math.abs((metaTheta ?? thetaCount) - videoCount) > 5;

  return (
    <div style={{
      width: '100vw', height: '100vh', position: 'relative', overflow: 'hidden',
      background: 'radial-gradient(circle at 50% 40%, #1e293b 0%, #0f172a 100%)',
      fontFamily: '"Inter", "Segoe UI", sans-serif',
    }}>

      {/* ── Sidebar ────────────────────────────────────────────────────────── */}
      <div style={{
        position: 'absolute', top: 20, left: 20, zIndex: 20,
        background: 'rgba(15,23,42,0.88)', backdropFilter: 'blur(14px)',
        border: '1px solid rgba(255,255,255,0.08)', borderRadius: 16,
        padding: '24px 20px', width: 280, color: 'white',
        boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
        maxHeight: 'calc(100vh - 40px)', overflowY: 'auto',
      }}>
        <h1 style={{ margin: '0 0 4px', fontSize: 22, fontWeight: 800, background: 'linear-gradient(90deg,#60a5fa,#c084fc)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          SMPL Kinematics
        </h1>
        <p style={{ margin: '0 0 16px', fontSize: 12, color: '#64748b' }}>
          {dataReady
            ? `${motionData.length} frames · ${playbackFps} fps${contactData.length > 0 ? ' · contact ✓' : ''}`
            : 'Loading…'}
        </p>

        {/* New Video */}
        {onNewVideo && (
          <>
            <button
              onClick={onNewVideo}
              style={{ width: '100%', marginBottom: 20, padding: '8px 0', borderRadius: 8, border: '1px solid rgba(255,255,255,0.1)', background: 'transparent', color: '#94a3b8', fontSize: 12, cursor: 'pointer', fontWeight: 600 }}
            >
              ← New video
            </button>
          </>
        )}

        {/* View Mode */}
        <label style={labelStyle}>View Mode</label>
        <div style={{ display: 'flex', gap: 6, marginBottom: 20 }}>
          {[{ id: '3d', label: '3D Only' }, { id: 'split', label: 'Video vs 3D', disabled: !videoUrl }].map(({ id, label, disabled }) => (
            <button key={id}
              onClick={() => !disabled && setViewMode(id)}
              disabled={!!disabled}
              style={{
                flex: 1, padding: '7px 0', borderRadius: 7, fontSize: 11, fontWeight: 700,
                cursor: disabled ? 'not-allowed' : 'pointer', border: 'none',
                background: viewMode === id ? 'linear-gradient(90deg,#3b82f6,#8b5cf6)' : '#1e293b',
                color: viewMode === id ? 'white' : disabled ? '#334155' : '#64748b',
                opacity: disabled ? 0.5 : 1,
              }}
            >{label}</button>
          ))}
        </div>

        {/* Body Model */}
        <label style={labelStyle}>Body Model</label>
        <select
          value={gender}
          onChange={e => { setGender(e.target.value); setCurrentFrame(0); }}
          style={{ width: '100%', marginBottom: 20, padding: '9px 12px', borderRadius: 8, background: '#0f172a', color: 'white', border: '1px solid #334155', fontSize: 14, cursor: 'pointer', outline: 'none' }}
        >
          <option value="female">Female</option>
          <option value="male">Male</option>
        </select>

        {/* Camera Mode */}
        <label style={labelStyle}>Camera Mode</label>
        <div style={{ display: 'flex', gap: 6, marginBottom: 20 }}>
          {['free', 'follow'].map(mode => (
            <button key={mode}
              onClick={() => setCameraMode(mode)}
              style={{
                flex: 1, padding: '7px 0', borderRadius: 7, fontSize: 11, fontWeight: 700,
                textTransform: 'uppercase', cursor: 'pointer', border: 'none',
                background: cameraMode === mode ? 'linear-gradient(90deg,#3b82f6,#8b5cf6)' : '#1e293b',
                color: cameraMode === mode ? 'white' : '#64748b',
              }}
            >
              {mode === 'follow' ? '🎯 Follow' : '🖱 Free'}
            </button>
          ))}
        </div>

        {/* Play/Pause */}
        <button
          onClick={() => setIsPlaying(p => !p)}
          disabled={!dataReady}
          style={{
            width: '100%', marginBottom: 14, padding: '10px 0', borderRadius: 8,
            background: isPlaying ? '#334155' : 'linear-gradient(90deg,#3b82f6,#8b5cf6)',
            color: 'white', border: 'none', fontWeight: 700, fontSize: 14,
            cursor: dataReady ? 'pointer' : 'not-allowed', transition: 'background .2s',
          }}
        >
          {isPlaying ? '⏸  Pause' : '▶  Play'}
        </button>

        {/* Scrubber */}
        <input
          type="range" min={0} max={totalFrames} value={currentFrame}
          onChange={e => { setIsPlaying(false); setCurrentFrame(Number(e.target.value)); }}
          style={{ width: '100%', marginBottom: 6, accentColor: '#60a5fa' }}
        />
        <div style={{ fontSize: 11, color: '#475569', textAlign: 'right', marginBottom: 16 }}>
          Frame {currentFrame} / {totalFrames}
          {playbackFps > 0 && <span style={{ marginLeft: 6 }}>· {(currentFrame / playbackFps).toFixed(2)}s</span>}
        </div>

        {/* Video offset (split mode only) */}
        {inSplitMode && (
          <>
            <label style={labelStyle}>Video Offset Frames</label>
            <input type="number" value={videoOffsetFrames}
              onChange={e => setVideoOffsetFrames(Number(e.target.value))}
              style={{ width: '100%', marginBottom: 16, padding: '8px 12px', borderRadius: 8, background: '#0f172a', color: 'white', border: '1px solid #334155', fontSize: 13, outline: 'none', boxSizing: 'border-box' }}
            />
          </>
        )}

        {/* Sync status */}
        <div style={{ marginBottom: 16, padding: '10px 12px', background: 'rgba(255,255,255,0.04)', borderRadius: 8, fontSize: 11, color: '#94a3b8' }}>
          <div style={{ marginBottom: 6, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1 }}>Sync Status</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            <span>FPS: <strong style={{ color: '#e2e8f0' }}>{playbackFps}</strong></span>
            <span>Theta frames: <strong style={{ color: '#e2e8f0' }}>{thetaCount}</strong></span>
            {videoCount !== null && <span>Video frames: <strong style={{ color: '#e2e8f0' }}>{videoCount}</strong></span>}
            {frameMismatch && <span style={{ color: '#fbbf24', marginTop: 2 }}>⚠ Frame count mismatch</span>}
          </div>
        </div>

        {/* Contact legend */}
        {contactData.length > 0 && (
          <div style={{ marginBottom: 16, padding: '10px 12px', background: 'rgba(255,255,255,0.04)', borderRadius: 8, fontSize: 11, color: '#94a3b8' }}>
            <div style={{ marginBottom: 4, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1 }}>Ground Contact</div>
            <div style={{ display: 'flex', gap: 12 }}>
              <span>🔵 Left foot</span>
              <span>🟣 Right foot</span>
            </div>
          </div>
        )}

        {/* Load warnings */}
        {loadWarnings.length > 0 && (
          <div style={{ marginBottom: 16, padding: '8px 12px', background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.25)', borderRadius: 8 }}>
            {loadWarnings.map((w, i) => (
              <div key={i} style={{ fontSize: 11, color: '#fbbf24', marginBottom: i < loadWarnings.length - 1 ? 4 : 0 }}>⚠ {w}</div>
            ))}
          </div>
        )}

        {/* Export */}
        <div style={{ borderTop: '1px solid rgba(255,255,255,0.07)', paddingTop: 16 }}>
          <label style={labelStyle}>Export</label>
          <button
            onClick={isRecording ? undefined : startRecording}
            disabled={!dataReady || isRecording}
            style={{
              width: '100%', padding: '10px 0', borderRadius: 8, border: 'none',
              fontWeight: 700, fontSize: 13, cursor: dataReady && !isRecording ? 'pointer' : 'not-allowed',
              transition: 'all .25s',
              background: isRecording ? 'rgba(239,68,68,0.15)' : dataReady ? 'linear-gradient(90deg,#10b981,#059669)' : '#1e293b',
              color: isRecording ? '#f87171' : 'white',
              boxShadow: isRecording ? 'none' : dataReady ? '0 0 16px rgba(16,185,129,0.35)' : 'none',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
            }}
          >
            {isRecording ? (
              <><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: '#f87171', animation: 'recPulse 0.9s ease-in-out infinite' }} />Recording… {recProgress}%</>
            ) : (
              <><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>Download Animation</>
            )}
          </button>
          {isRecording && (
            <div style={{ marginTop: 8, height: 4, borderRadius: 4, background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
              <div style={{ height: '100%', borderRadius: 4, background: 'linear-gradient(90deg,#f87171,#ef4444)', width: `${recProgress}%`, transition: 'width 0.15s linear' }} />
            </div>
          )}
        </div>
      </div>

      {/* ── Video panel (split mode) ────────────────────────────────────────── */}
      {/* Always mount when videoUrl is known so videoRef is ready for seeking */}
      {videoUrl && (
        <div style={{ position: 'absolute', top: 0, left: 0, zIndex: 1, width: '50%', height: '100%', background: '#000', overflow: 'hidden', display: inSplitMode ? 'block' : 'none' }}>
          <video ref={videoRef} src={videoUrl} muted preload="auto" playsInline
            style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }} />
          <div style={{ position: 'absolute', top: 12, right: 12, background: 'rgba(0,0,0,0.55)', borderRadius: 6, padding: '4px 10px', fontSize: 11, color: '#94a3b8', fontWeight: 700, pointerEvents: 'none' }}>
            Input Video
          </div>
          <div style={{ position: 'absolute', bottom: 12, left: 12, background: 'rgba(0,0,0,0.55)', borderRadius: 6, padding: '5px 10px', fontSize: 11, color: '#cbd5e1', display: 'flex', gap: 12, pointerEvents: 'none' }}>
            {(() => { const vf = Math.max(0, Math.min(currentFrame + videoOffsetFrames, totalFrames)); return (<><span>Frame {vf} / {videoCount ?? totalFrames}</span><span>{(vf / playbackFps).toFixed(2)}s</span><span>{playbackFps} fps</span></>); })()}
          </div>
        </div>
      )}

      {/* ── 3D Canvas ───────────────────────────────────────────────────────── */}
      <div style={{
        position: 'absolute', top: 0, zIndex: 0,
        left: inSplitMode ? '50%' : '0',
        width: inSplitMode ? '50%' : '100%',
        height: '100%',
        transition: 'left 0.15s ease, width 0.15s ease',
      }}>
        <Canvas camera={{ position: [0, 1.0, 4], fov: 50 }} shadows
          gl={{ antialias: true, preserveDrawingBuffer: true }}
          style={{ width: '100%', height: '100%' }}>
          <color attach="background" args={['#0f172a']} />
          <ambientLight intensity={0.5} />
          <spotLight position={[4, 8, 4]}   angle={0.3} penumbra={0.8} intensity={2.5} castShadow shadow-mapSize={1024} />
          <spotLight position={[-4, 6, -4]} angle={0.5} penumbra={1}   intensity={1.0} color="#93c5fd" />

          <Environment preset="city" />

          <Suspense fallback={null}>
            {dataReady && (
              <SMPLAnimator gender={gender} motionData={motionData} currentFrame={currentFrame}
                betaData={betaData} transData={transData} contactData={contactData}
                onGroundY={setGroundY} groupRef={avatarGroupRef}
              />
            )}
          </Suspense>

          {cameraMode === 'follow' && dataReady && (
            <CameraFollow avatarGroupRef={avatarGroupRef} controlsRef={controlsRef} />
          )}

          <ContactShadows position={[0, groundY, 0]} opacity={0.55} scale={20} blur={2} far={4} />
          <gridHelper args={[40, 40, '#1e293b', '#1e293b']} position={[0, groundY, 0]} />

          <OrbitControls ref={controlsRef}
            minDistance={1.5} maxDistance={20}
            maxPolarAngle={Math.PI / 2 + 0.1}
            enableDamping={cameraMode === 'free'} dampingFactor={0.06}
          />

          <CanvasRecorder isRecording={isRecording} onFrameCaptured={handleRecordFrame} onStop={handleRecordStop} />
        </Canvas>

        {inSplitMode && (
          <div style={{ position: 'absolute', top: 12, left: 12, zIndex: 5, background: 'rgba(0,0,0,0.55)', borderRadius: 6, padding: '4px 10px', fontSize: 11, color: '#94a3b8', fontWeight: 700, pointerEvents: 'none' }}>
            Reconstruction
          </div>
        )}
      </div>

      <style>{`
        @keyframes recPulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.7); } }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
      `}</style>
    </div>
  );
}

// ============================================================================
// Root — screen state machine
// ============================================================================
export default function App() {
  const [screen, setScreen] = useState('upload');  // 'upload' | 'processing' | 'viewer'
  const [jobId,  setJobId]  = useState(null);
  const [error,  setError]  = useState(null);

  if (error) {
    return (
      <div style={{ ...s.page }}>
        <div style={s.card}>
          <h2 style={{ ...s.title, color: '#f87171' }}>Pipeline Error</h2>
          <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 24, lineHeight: 1.7 }}>{error}</p>
          <button onClick={() => { setError(null); setScreen('upload'); setJobId(null); }}
            style={{ padding: '10px 28px', borderRadius: 8, background: 'linear-gradient(90deg,#3b82f6,#8b5cf6)', color: 'white', border: 'none', fontWeight: 700, cursor: 'pointer' }}>
            Try again
          </button>
        </div>
      </div>
    );
  }

  if (screen === 'upload')
    return <UploadScreen onUpload={(id) => { setJobId(id); setScreen('processing'); }} />;

  if (screen === 'processing')
    return (
      <ProcessingScreen
        jobId={jobId}
        onDone={() => setScreen('viewer')}
        onError={(msg) => setError(msg)}
      />
    );

  return (
    <ViewerApp
      dataBaseUrl={`/api/results/${jobId}`}
      onNewVideo={() => { setScreen('upload'); setJobId(null); }}
    />
  );
}

// ── Shared style constants ─────────────────────────────────────────────────
const s = {
  page: {
    width: '100vw', height: '100vh', display: 'flex', alignItems: 'center',
    justifyContent: 'center',
    background: 'radial-gradient(circle at 50% 40%, #1e293b 0%, #0f172a 100%)',
    fontFamily: '"Inter", "Segoe UI", sans-serif',
  },
  card: {
    background: 'rgba(15,23,42,0.88)', backdropFilter: 'blur(20px)',
    border: '1px solid rgba(255,255,255,0.08)', borderRadius: 20,
    padding: '40px 36px', width: '100%', maxWidth: 460,
    boxShadow: '0 24px 64px rgba(0,0,0,0.6)',
    display: 'flex', flexDirection: 'column', alignItems: 'center',
    color: 'white',
  },
  logo: { display: 'flex', gap: 2, marginBottom: 16 },
  logoW: { fontSize: 36, fontWeight: 900, color: '#60a5fa' },
  logoH: { fontSize: 36, fontWeight: 900, color: '#818cf8' },
  logoA: { fontSize: 36, fontWeight: 900, color: '#a78bfa' },
  logoM: { fontSize: 36, fontWeight: 900, color: '#c084fc' },
  title: { margin: '0 0 8px', fontSize: 26, fontWeight: 800, color: 'white', textAlign: 'center' },
  subtitle: { margin: '0 0 28px', fontSize: 13, color: '#64748b', textAlign: 'center', lineHeight: 1.6 },
  dropzone: {
    width: '100%', border: '2px dashed rgba(99,102,241,0.4)',
    borderRadius: 12, padding: '36px 20px', marginBottom: 20,
    display: 'flex', flexDirection: 'column', alignItems: 'center',
    cursor: 'pointer', transition: 'border-color 0.2s, background 0.2s',
    background: 'rgba(99,102,241,0.04)',
    boxSizing: 'border-box',
  },
  dropzoneActive: {
    borderColor: '#60a5fa', background: 'rgba(99,102,241,0.12)',
  },
  dropText: { margin: '0 0 4px', fontSize: 14, color: '#94a3b8', textAlign: 'center' },
  dropHint: { margin: 0, fontSize: 11, color: '#475569' },
  errorBox: {
    width: '100%', padding: '10px 14px', marginBottom: 16,
    background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
    borderRadius: 8, fontSize: 12, color: '#f87171',
    boxSizing: 'border-box',
  },
  infoRow: {
    display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center',
    marginTop: 4,
  },
  infoItem: { fontSize: 10, color: '#475569', padding: '4px 8px', background: 'rgba(255,255,255,0.04)', borderRadius: 6 },
  spinner: {
    width: 32, height: 32, borderRadius: '50%', marginBottom: 12,
    border: '3px solid rgba(99,102,241,0.2)',
    borderTopColor: '#60a5fa',
    animation: 'spin 0.8s linear infinite',
  },
};

const labelStyle = {
  display: 'block', marginBottom: 6, fontSize: 11,
  fontWeight: 700, color: '#94a3b8',
  textTransform: 'uppercase', letterSpacing: 1,
};
