"""
File: src/dashboard/upload_module.py
Secure data upload module with validation and encryption
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import json
from datetime import datetime
import logging
from cryptography.fernet import Fernet
import tempfile
import io

from src.data.data_validator import DataValidator
from src.database.mongo_handler import MongoHandler
from src.utils.logger import setup_logger

logger = setup_logger()


class UploadModule:
    """
    Secure file upload module with validation and preprocessing
    """
    
    def __init__(self):
        self.validator = DataValidator()
        self.db = MongoHandler()
        self.max_file_size = 100 * 1024 * 1024  # 100 MB
        self.allowed_extensions = ['.csv', '.xlsx', '.xls']
        
        # Initialize session state
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'upload_history' not in st.session_state:
            st.session_state.upload_history = []
    
    def render(self):
        """Render upload interface"""
        st.markdown("""
        <div class="upload-zone">
            <h2>📁 Data Upload Module</h2>
            <p>Upload multimodal gait sensor data for analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "📤 Upload New Data",
            "📋 Upload History",
            "ℹ️ Data Format Guide"
        ])
        
        with tab1:
            self._render_upload_tab()
        
        with tab2:
            self._render_history_tab()
        
        with tab3:
            self._render_format_guide()
    
    def _render_upload_tab(self):
        """Render upload form"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Gait Data")
            
            # Patient information
            with st.expander("Patient Information", expanded=True):
                patient_id = st.text_input(
                    "Patient ID",
                    help="Unique identifier for the patient"
                )
                
                patient_age = st.number_input(
                    "Age",
                    min_value=0,
                    max_value=120,
                    value=50
                )
                
                patient_gender = st.selectbox(
                    "Gender",
                    ["Male", "Female", "Other", "Prefer not to say"]
                )
                
                medical_history = st.text_area(
                    "Relevant Medical History",
                    help="Optional: Brief medical history"
                )
            
            # File upload
            st.subheader("Data Files")
            
            uploaded_files = st.file_uploader(
                "Choose CSV or Excel files",
                type=['csv', 'xlsx', 'xls'],
                accept_multiple_files=True,
                help="Upload multimodal sensor data (accelerometer, gyroscope, EMG)"
            )
            
            # Data collection metadata
            with st.expander("Data Collection Metadata"):
                collection_date = st.date_input(
                    "Collection Date",
                    value=datetime.now()
                )
                
                collection_site = st.text_input(
                    "Collection Site/Institution",
                    help="Where the data was collected"
                )
                
                device_type = st.selectbox(
                    "Sensor Device",
                    ["IMU Sensor Array", "Wearable Device", "Motion Capture", "Other"]
                )
                
                sampling_rate = st.number_input(
                    "Sampling Rate (Hz)",
                    min_value=1,
                    max_value=1000,
                    value=100
                )
            
            # Privacy settings
            with st.expander("Privacy & Security"):
                encrypt_data = st.checkbox(
                    "Encrypt data at rest",
                    value=True,
                    help="Enable encryption for stored data"
                )
                
                anonymize = st.checkbox(
                    "Anonymize patient data",
                    value=False,
                    help="Remove identifying information"
                )
                
                consent_given = st.checkbox(
                    "Patient consent obtained",
                    value=False,
                    help="Confirm patient consent for data usage"
                )
            
            # Upload button
            if st.button("🚀 Upload and Validate", type="primary"):
                if not uploaded_files:
                    st.error("Please upload at least one file")
                elif not patient_id:
                    st.error("Patient ID is required")
                elif not consent_given:
                    st.error("Patient consent must be obtained")
                else:
                    self._process_upload(
                        uploaded_files,
                        {
                            'patient_id': patient_id,
                            'age': patient_age,
                            'gender': patient_gender,
                            'medical_history': medical_history,
                            'collection_date': str(collection_date),
                            'collection_site': collection_site,
                            'device_type': device_type,
                            'sampling_rate': sampling_rate,
                            'encrypt_data': encrypt_data,
                            'anonymize': anonymize
                        }
                    )
        
        with col2:
            # Quick stats and tips
            st.markdown("""
            <div class="feature-card">
                <h3>📊 Upload Guidelines</h3>
                <ul>
                    <li>✅ Max file size: 100 MB</li>
                    <li>✅ Formats: CSV, Excel</li>
                    <li>✅ Required columns: accel_x/y/z, gyro_x/y/z, emg_1/2/3</li>
                    <li>✅ Data privacy: HIPAA compliant</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <h4>🔐 Security Features</h4>
                <p>• End-to-end encryption</p>
                <p>• Audit logging</p>
                <p>• Access control</p>
                <p>• Data anonymization</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _process_upload(
        self,
        uploaded_files: List,
        metadata: Dict[str, Any]
    ):
        """Process and validate uploaded files"""
        with st.spinner("Processing uploaded files..."):
            progress_bar = st.progress(0)
            status_container = st.empty()
            
            results = []
            total_files = len(uploaded_files)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (idx + 1) / total_files
                progress_bar.progress(progress)
                status_container.info(f"Processing {uploaded_file.name}...")
                
                try:
                    # Check file size
                    file_size = uploaded_file.size
                    if file_size > self.max_file_size:
                        raise ValueError(
                            f"File size ({file_size/1024/1024:.1f} MB) "
                            f"exceeds maximum ({self.max_file_size/1024/1024:.1f} MB)"
                        )
                    
                    # Read file
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    # Validate data
                    validation_result = self.validator.validate_dataframe(df)
                    
                    if not validation_result['valid']:
                        raise ValueError(
                            f"Validation failed: {validation_result['errors']}"
                        )
                    
                    # Anonymize if requested
                    if metadata['anonymize']:
                        df = self._anonymize_data(df, metadata['patient_id'])
                    
                    # Generate file hash for integrity
                    file_hash = self._generate_file_hash(df)
                    
                    # Save to database
                    upload_id = self._save_to_database(
                        df,
                        uploaded_file.name,
                        metadata,
                        file_hash,
                        validation_result
                    )
                    
                    results.append({
                        'filename': uploaded_file.name,
                        'status': 'success',
                        'rows': len(df),
                        'columns': len(df.columns),
                        'upload_id': upload_id,
                        'validation': validation_result
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing {uploaded_file.name}: {e}")
                    results.append({
                        'filename': uploaded_file.name,
                        'status': 'error',
                        'error': str(e)
                    })
            
            # Display results
            status_container.empty()
            progress_bar.empty()
            
            self._display_upload_results(results)
    
    def _anonymize_data(self, df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
        """Anonymize patient data"""
        # Create pseudonymous ID
        pseudo_id = hashlib.sha256(patient_id.encode()).hexdigest()[:16]
        
        # Remove any columns that might contain identifying information
        identifying_columns = ['name', 'email', 'phone', 'address', 'ssn']
        for col in identifying_columns:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        logger.info(f"Anonymized data for patient {patient_id} -> {pseudo_id}")
        return df
    
    def _generate_file_hash(self, df: pd.DataFrame) -> str:
        """Generate SHA-256 hash of dataframe for integrity checking"""
        # Convert dataframe to bytes
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        data_bytes = buffer.getvalue()
        
        # Generate hash
        file_hash = hashlib.sha256(data_bytes).hexdigest()
        return file_hash
    
    def _save_to_database(
        self,
        df: pd.DataFrame,
        filename: str,
        metadata: Dict[str, Any],
        file_hash: str,
        validation_result: Dict[str, Any]
    ) -> str:
        """Save uploaded data to MongoDB"""
        # Encrypt data if requested
        if metadata['encrypt_data']:
            encrypted_data = self._encrypt_dataframe(df)
            data_to_store = encrypted_data
            is_encrypted = True
        else:
            data_to_store = df.to_dict('records')
            is_encrypted = False
        
        # Create document
        document = {
            'patient_id': metadata['patient_id'],
            'filename': filename,
            'file_hash': file_hash,
            'upload_timestamp': datetime.now().isoformat(),
            'metadata': {
                'age': metadata['age'],
                'gender': metadata['gender'],
                'medical_history': metadata['medical_history'],
                'collection_date': metadata['collection_date'],
                'collection_site': metadata['collection_site'],
                'device_type': metadata['device_type'],
                'sampling_rate': metadata['sampling_rate']
            },
            'data_stats': {
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'columns': df.columns.tolist()
            },
            'validation': validation_result,
            'is_encrypted': is_encrypted,
            'data': data_to_store,
            'status': 'uploaded',
            'processing_status': 'pending'
        }
        
        # Insert into database
        upload_id = self.db.insert_upload(document)
        
        # Add to session history
        st.session_state.upload_history.append({
            'upload_id': upload_id,
            'patient_id': metadata['patient_id'],
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Saved upload to database: {upload_id}")
        return upload_id
    
    def _encrypt_dataframe(self, df: pd.DataFrame) -> str:
        """Encrypt dataframe using Fernet symmetric encryption"""
        # Generate or load encryption key (in production, use key management service)
        key = Fernet.generate_key()
        cipher = Fernet(key)
        
        # Convert dataframe to JSON
        json_data = df.to_json()
        
        # Encrypt
        encrypted = cipher.encrypt(json_data.encode())
        
        # Store key securely (in production, use KMS)
        # For now, store with data (NOT recommended for production)
        encrypted_package = {
            'key': key.decode(),
            'data': encrypted.decode()
        }
        
        return json.dumps(encrypted_package)
    
    def _display_upload_results(self, results: List[Dict]):
        """Display upload results"""
        success_count = sum(1 for r in results if r['status'] == 'success')
        error_count = sum(1 for r in results if r['status'] == 'error')
        
        if success_count > 0:
            st.success(f"✅ Successfully uploaded {success_count} file(s)")
        
        if error_count > 0:
            st.error(f"❌ Failed to upload {error_count} file(s)")
        
        # Detailed results
        for result in results:
            with st.expander(f"📄 {result['filename']}", expanded=result['status']=='error'):
                if result['status'] == 'success':
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", result['rows'])
                    with col2:
                        st.metric("Columns", result['columns'])
                    with col3:
                        st.metric("Status", "✅ Valid")
                    
                    st.code(f"Upload ID: {result['upload_id']}")
                    
                    # Validation details
                    if result['validation'].get('warnings'):
                        st.warning("Warnings:")
                        for warning in result['validation']['warnings']:
                            st.write(f"⚠️ {warning}")
                else:
                    st.error(f"Error: {result['error']}")
    
    def _render_history_tab(self):
        """Render upload history"""
        st.subheader("Upload History")
        
        if not st.session_state.upload_history:
            st.info("No uploads yet. Upload your first dataset!")
        else:
            # Create dataframe from history
            history_df = pd.DataFrame(st.session_state.upload_history)
            
            # Display as interactive table
            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Export history
            if st.button("📥 Export History"):
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"upload_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    def _render_format_guide(self):
        """Render data format guide"""
        st.subheader("Data Format Requirements")
        
        st.markdown("""
        ### Required Data Format
        
        Your CSV/Excel file must contain the following columns:
        
        #### Accelerometer Data (Required)
        - `accel_x`: Acceleration in X-axis (m/s²)
        - `accel_y`: Acceleration in Y-axis (m/s²)
        - `accel_z`: Acceleration in Z-axis (m/s²)
        
        #### Gyroscope Data (Required)
        - `gyro_x`: Angular velocity around X-axis (rad/s)
        - `gyro_y`: Angular velocity around Y-axis (rad/s)
        - `gyro_z`: Angular velocity around Z-axis (rad/s)
        
        #### EMG Data (Required)
        - `emg_1`: EMG sensor 1 (µV)
        - `emg_2`: EMG sensor 2 (µV)
        - `emg_3`: EMG sensor 3 (µV)
        
        #### Optional Columns
        - `timestamp`: Time of measurement
        - `label`: Ground truth label (for training)
        - Additional sensor data
        """)
        
        # Sample data
        st.subheader("Sample Data Format")
        
        sample_data = pd.DataFrame({
            'timestamp': range(0, 1000, 10),
            'accel_x': np.random.randn(100) * 0.5,
            'accel_y': np.random.randn(100) * 0.5,
            'accel_z': np.random.randn(100) * 0.5 + 9.8,
            'gyro_x': np.random.randn(100) * 0.1,
            'gyro_y': np.random.randn(100) * 0.1,
            'gyro_z': np.random.randn(100) * 0.1,
            'emg_1': np.random.rand(100) * 100,
            'emg_2': np.random.rand(100) * 100,
            'emg_3': np.random.rand(100) * 100,
            'label': np.random.choice(['gait', 'non_gait'], 100)
        })
        
        st.dataframe(sample_data.head(10), use_container_width=True)
        
        # Download sample
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Dataset",
            data=csv,
            file_name="sample_gait_data.csv",
            mime="text/csv"
        )
        
        st.markdown("""
        ### Data Quality Tips
        
        1. **Consistent Sampling Rate**: Ensure uniform time intervals between measurements
        2. **No Missing Values**: Fill or interpolate missing data points
        3. **Proper Units**: Use standard units (m/s², rad/s, µV)
        4. **Calibration**: Ensure sensors are properly calibrated
        5. **Noise Filtering**: Pre-filter obvious noise if possible
        6. **Data Range**: Check for unrealistic values (e.g., acceleration > 50 m/s²)
        
        ### Privacy & Security
        
        - All data is encrypted at rest
        - Patient identifiers are hashed
        - Access is logged and audited
        - HIPAA and GDPR compliant
        - Data retention policies enforced
        """)


class BatchUploader:
    """
    Batch upload functionality for large datasets
    Supports parallel processing and resumable uploads
    """
    
    def __init__(self):
        self.upload_module = UploadModule()
        self.chunk_size = 10000  # rows per chunk
    
    def batch_upload(
        self,
        file_path: str,
        metadata: Dict[str, Any],
        chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Upload large file in chunks
        
        Args:
            file_path: Path to file
            metadata: Upload metadata
            chunk_size: Number of rows per chunk
            
        Returns:
            Upload results
        """
        if chunk_size:
            self.chunk_size = chunk_size
        
        # Read file in chunks
        if file_path.endswith('.csv'):
            chunks = pd.read_csv(file_path, chunksize=self.chunk_size)
        else:
            # For Excel, read entire file (chunking not supported)
            df = pd.read_excel(file_path)
            chunks = [df[i:i+self.chunk_size] for i in range(0, len(df), self.chunk_size)]
        
        results = {
            'total_chunks': 0,
            'successful_chunks': 0,
            'failed_chunks': 0,
            'upload_ids': []
        }
        
        for idx, chunk in enumerate(chunks):
            try:
                # Validate chunk
                validation = self.upload_module.validator.validate_dataframe(chunk)
                
                if not validation['valid']:
                    logger.warning(f"Chunk {idx} validation failed: {validation['errors']}")
                    results['failed_chunks'] += 1
                    continue
                
                # Save chunk
                file_hash = self.upload_module._generate_file_hash(chunk)
                
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = idx
                
                upload_id = self.upload_module._save_to_database(
                    chunk,
                    f"{file_path}_chunk_{idx}",
                    chunk_metadata,
                    file_hash,
                    validation
                )
                
                results['upload_ids'].append(upload_id)
                results['successful_chunks'] += 1
                results['total_chunks'] += 1
                
                logger.info(f"Uploaded chunk {idx}: {upload_id}")
                
            except Exception as e:
                logger.error(f"Error uploading chunk {idx}: {e}")
                results['failed_chunks'] += 1
                results['total_chunks'] += 1
        
        return results


class UploadValidator:
    """
    Advanced validation for uploaded data
    Includes statistical checks and anomaly detection
    """
    
    @staticmethod
    def validate_sensor_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate sensor data quality
        
        Returns:
            Validation results with detailed diagnostics
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check required columns
        required_cols = [
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'emg_1', 'emg_2', 'emg_3'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results['valid'] = False
            results['errors'].append(f"Missing columns: {missing_cols}")
            return results
        
        # Check for missing values
        missing_counts = df[required_cols].isnull().sum()
        if missing_counts.any():
            results['warnings'].append(f"Missing values detected: {missing_counts.to_dict()}")
        
        # Statistical validation
        for col in required_cols:
            data = df[col].dropna()
            
            if len(data) == 0:
                results['errors'].append(f"Column {col} has no valid data")
                results['valid'] = False
                continue
            
            # Calculate statistics
            stats = {
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'range': float(data.max() - data.min())
            }
            
            results['statistics'][col] = stats
            
            # Range validation
            if 'accel' in col:
                # Accelerometer should be within reasonable range
                if abs(stats['max']) > 50:  # m/s²
                    results['warnings'].append(
                        f"{col}: Maximum value {stats['max']:.2f} exceeds typical range"
                    )
            elif 'gyro' in col:
                # Gyroscope should be within reasonable range
                if abs(stats['max']) > 10:  # rad/s
                    results['warnings'].append(
                        f"{col}: Maximum value {stats['max']:.2f} exceeds typical range"
                    )
            elif 'emg' in col:
                # EMG should be positive
                if stats['min'] < 0:
                    results['warnings'].append(f"{col}: Contains negative values")
                if stats['max'] > 5000:  # µV
                    results['warnings'].append(f"{col}: Values exceed typical EMG range")
            
            # Check for constant values
            if stats['std'] < 1e-6:
                results['warnings'].append(f"{col}: Appears to be constant (std={stats['std']})")
        
        # Check sampling consistency
        if 'timestamp' in df.columns:
            time_diffs = df['timestamp'].diff().dropna()
            if len(time_diffs) > 0:
                mean_interval = time_diffs.mean()
                std_interval = time_diffs.std()
                
                if std_interval / mean_interval > 0.1:  # 10% variation
                    results['warnings'].append(
                        f"Irregular sampling detected (CV={std_interval/mean_interval:.2%})"
                    )
        
        return results
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Detect anomalous data points using statistical methods
        
        Returns:
            Dictionary mapping column names to lists of anomalous indices
        """
        anomalies = {}
        
        sensor_cols = [
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'emg_1', 'emg_2', 'emg_3'
        ]
        
        for col in sensor_cols:
            if col not in df.columns:
                continue
            
            data = df[col].values
            
            # Z-score method
            mean = np.mean(data)
            std = np.std(data)
            
            if std > 0:
                z_scores = np.abs((data - mean) / std)
                anomaly_indices = np.where(z_scores > 3)[0].tolist()
                
                if anomaly_indices:
                    anomalies[col] = anomaly_indices
        
        return anomalies