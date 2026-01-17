// src/types.ts

export interface User {
    id: string;
    username: string;
    email?: string;
}

export interface Session {
    _id: string;
    status: 'idle' | 'running' | 'completed' | 'failed';
    num_peers: number;
    created_at: string;
    updated_at?: string;
    model_type?: string;
    dataset?: string;
}

export interface EpochData {
    epoch: number;
    loss: number;
    accuracy: number;
    val_loss?: number;
    val_accuracy?: number;
    timestamp?: string;
}

export interface Hyperparameters {
    epochs: number;
    batch_size: number;
    lr?: number;
    learning_rate?: number;
    optimizer?: string;
}

export interface PeerData {
    peer_id: string;
    status: 'active' | 'idle' | 'disconnected';
    epochs: EpochData[];
    hyperparameters: Hyperparameters;
    current_epoch?: number;
    total_epochs?: number;
    device?: string;
    last_updated?: string;
}

export interface FullResultsResponse {
    session_id: string;
    status: string;
    peers: {
        [peerId: string]: PeerData;
    };
    global_metrics?: {
        avg_loss: number;
        avg_accuracy: number;
        total_epochs_completed: number;
    };
    created_at?: string;
    updated_at?: string;
}

export interface ApiResponse<T> {
    success: boolean;
    data?: T;
    error?: string;
    message?: string;
}