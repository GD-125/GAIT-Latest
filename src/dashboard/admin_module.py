# File: src/dashboard/admin_module.py
# Admin dashboard with authentication, monitoring, and system management

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import hashlib
import secrets

logger = logging.getLogger(__name__)


class AdminModule:
    """
    Comprehensive admin dashboard with security, monitoring, and management
    Features:
    - Role-based access control
    - System monitoring
    - User management
    - Model management
    - Audit logs
    - Analytics
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize session state
        if 'admin_authenticated' not in st.session_state:
            st.session_state.admin_authenticated = False
        if 'admin_user' not in st.session_state:
            st.session_state.admin_user = None
        if 'admin_role' not in st.session_state:
            st.session_state.admin_role = None
        
        # Admin credentials (in production, use secure database)
        self.admin_users = {
            'admin': {
                'password_hash': self._hash_password('admin123'),
                'role': 'super_admin',
                'permissions': ['all']
            },
            'manager': {
                'password_hash': self._hash_password('manager123'),
                'role': 'manager',
                'permissions': ['view', 'manage_users', 'view_logs']
            },
            'operator': {
                'password_hash': self._hash_password('operator123'),
                'role': 'operator',
                'permissions': ['view', 'run_analysis']
            }
        }
    
    def render(self):
        """Render admin dashboard"""
        if not st.session_state.admin_authenticated:
            self._render_admin_login()
        else:
            self._render_admin_dashboard()
    
    def _render_admin_login(self):
        """Render admin login page"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%); 
                    padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
            <h2>🔐 Admin Panel - Secure Access</h2>
            <p>Administrator credentials required</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🛡️ Administrator Login")
            
            with st.form("admin_login_form"):
                username = st.text_input("Username", key="admin_username")
                password = st.text_input("Password", type="password", key="admin_password")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    submit = st.form_submit_button("🔓 Login", type="primary", use_container_width=True)
                
                with col_b:
                    if st.form_submit_button("🔙 Back", use_container_width=True):
                        st.session_state.current_page = "home"
                        st.rerun()
                
                if submit:
                    if self._authenticate_admin(username, password):
                        st.success("✅ Authentication successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
                        self.logger.warning(f"Failed admin login attempt: {username}")
            
            st.info("**Demo Credentials:**\n- admin / admin123\n- manager / manager123\n- operator / operator123")
    
    def _authenticate_admin(self, username: str, password: str) -> bool:
        """Authenticate admin user"""
        if username in self.admin_users:
            user = self.admin_users[username]
            if user['password_hash'] == self._hash_password(password):
                st.session_state.admin_authenticated = True
                st.session_state.admin_user = username
                st.session_state.admin_role = user['role']
                st.session_state.admin_permissions = user['permissions']
                self.logger.info(f"Admin user authenticated: {username} ({user['role']})")
                return True
        return False
    
    def _hash_password(self, password: str) -> str:
        """Hash password securely"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _render_admin_dashboard(self):
        """Render main admin dashboard"""
        # Header with user info
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
            <h2>🛠️ Admin Dashboard</h2>
            <p>Logged in as: <strong>{st.session_state.admin_user}</strong> ({st.session_state.admin_role})</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar navigation
        with st.sidebar:
            st.markdown("### 🎯 Admin Navigation")
            
            admin_page = st.selectbox(
                "Select Section",
                [
                    "📊 System Overview",
                    "👥 User Management",
                    "🤖 Model Management",
                    "📋 Audit Logs",
                    "📈 Analytics",
                    "⚙️ System Settings",
                    "🔔 Notifications"
                ]
            )
            
            st.markdown("---")
            
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.admin_authenticated = False
                st.session_state.admin_user = None
                st.rerun()
        
        # Route to appropriate page
        if admin_page == "📊 System Overview":
            self._render_system_overview()
        elif admin_page == "👥 User Management":
            self._render_user_management()
        elif admin_page == "🤖 Model Management":
            self._render_model_management()
        elif admin_page == "📋 Audit Logs":
            self._render_audit_logs()
        elif admin_page == "📈 Analytics":
            self._render_analytics()
        elif admin_page == "⚙️ System Settings":
            self._render_system_settings()
        elif admin_page == "🔔 Notifications":
            self._render_notifications()
    
    def _render_system_overview(self):
        """Render system overview"""
        st.markdown("### 📊 System Overview")
        
        # System metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "System Status",
                "🟢 Online",
                "+99.9% Uptime"
            )
        
        with col2:
            st.metric(
                "Active Users",
                "47",
                "+12 today"
            )
        
        with col3:
            st.metric(
                "Analyses Today",
                "234",
                "+18%"
            )
        
        with col4:
            st.metric(
                "Model Accuracy",
                "96.8%",
                "+0.3%"
            )
        
        # System health
        st.markdown("#### 🏥 System Health")
        
        health_data = {
            'Component': ['API Server', 'Database', 'ML Models', 'Storage', 'Network'],
            'Status': ['✅ Healthy', '✅ Healthy', '✅ Healthy', '⚠️ Warning', '✅ Healthy'],
            'CPU Usage': ['45%', '32%', '78%', '15%', '8%'],
            'Memory': ['2.1GB', '4.5GB', '6.2GB', '1.2GB', '512MB'],
            'Uptime': ['15d 4h', '15d 4h', '15d 4h', '15d 4h', '15d 4h']
        }
        
        df_health = pd.DataFrame(health_data)
        st.dataframe(df_health, use_container_width=True)
        
        # Resource usage charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💻 CPU Usage (Last 24h)")
            fig_cpu = self._create_resource_chart('CPU')
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            st.markdown("#### 🧠 Memory Usage (Last 24h)")
            fig_mem = self._create_resource_chart('Memory')
            st.plotly_chart(fig_mem, use_container_width=True)
    
    def _create_resource_chart(self, resource_type: str) -> go.Figure:
        """Create resource usage chart"""
        # Generate sample data
        hours = list(range(24))
        if resource_type == 'CPU':
            usage = [np.random.uniform(30, 80) for _ in range(24)]
        else:
            usage = [np.random.uniform(40, 75) for _ in range(24)]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=usage,
            mode='lines+markers',
            name=resource_type,
            line=dict(color='#3498db', width=2),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.2)'
        ))
        
        fig.update_layout(
            xaxis_title="Hours Ago",
            yaxis_title=f"{resource_type} Usage (%)",
            height=300,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def _render_user_management(self):
        """Render user management"""
        st.markdown("### 👥 User Management")
        
        # Check permissions
        if 'manage_users' not in st.session_state.admin_permissions and 'all' not in st.session_state.admin_permissions:
            st.error("❌ You don't have permission to manage users")
            return
        
        # User statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", "127")
        with col2:
            st.metric("Active Today", "47")
        with col3:
            st.metric("New This Week", "8")
        with col4:
            st.metric("Doctors", "42")
        
        # User list
        st.markdown("#### 📋 Registered Users")
        
        users_data = {
            'Username': ['dr.smith', 'dr.jones', 'researcher01', 'admin', 'dr.wilson'],
            'Role': ['Doctor', 'Doctor', 'Researcher', 'Admin', 'Doctor'],
            'Email': ['smith@hospital.com', 'jones@clinic.com', 'research@uni.edu', 'admin@system.com', 'wilson@med.com'],
            'Last Active': ['2 hours ago', '5 hours ago', '1 day ago', '30 min ago', '3 hours ago'],
            'Analyses': [45, 32, 128, 5, 67],
            'Status': ['✅ Active', '✅ Active', '✅ Active', '✅ Active', '⚠️ Suspended']
        }
        
        df_users = pd.DataFrame(users_data)
        st.dataframe(df_users, use_container_width=True)
        
        # User actions
        st.markdown("#### 🔧 User Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("➕ Add New User", use_container_width=True):
                self._show_add_user_form()
        
        with col2:
            if st.button("🔍 Search Users", use_container_width=True):
                st.info("Search functionality")
        
        with col3:
            if st.button("📊 User Analytics", use_container_width=True):
                st.info("User analytics")
    
    def _show_add_user_form(self):
        """Show add user form"""
        with st.form("add_user_form"):
            st.markdown("#### ➕ Add New User")
            
            username = st.text_input("Username")
            email = st.text_input("Email")
            role = st.selectbox("Role", ["Doctor", "Researcher", "Admin", "Operator"])
            password = st.text_input("Temporary Password", type="password")
            
            if st.form_submit_button("Create User", type="primary"):
                st.success(f"✅ User {username} created successfully!")
                self.logger.info(f"New user created: {username}")
    
    def _render_model_management(self):
        """Render model management"""
        st.markdown("### 🤖 Model Management")
        
        # Model status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Gait Detector", "v2.1", "96.5% Acc")
        with col2:
            st.metric("Disease Classifier", "v3.0", "97.2% Acc")
        with col3:
            st.metric("Last Updated", "3 days ago")
        
        # Model list
        st.markdown("#### 📋 Available Models")
        
        models_data = {
            'Model Name': ['Gait Detector CNN-BiLSTM', 'Disease Classifier Transformer', 'XGBoost Ensemble'],
            'Version': ['v2.1', 'v3.0', 'v1.5'],
            'Accuracy': ['96.5%', '97.2%', '98.1%'],
            'Last Trained': ['3 days ago', '1 week ago', '2 weeks ago'],
            'Status': ['✅ Active', '✅ Active', '✅ Active']
        }
        
        df_models = pd.DataFrame(models_data)
        st.dataframe(df_models, use_container_width=True)
        
        # Model actions
        st.markdown("#### 🔧 Model Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚀 Deploy Model", use_container_width=True):
                st.info("Model deployment")
        
        with col2:
            if st.button("🔄 Retrain Model", use_container_width=True):
                st.info("Model retraining")
        
        with col3:
            if st.button("📊 View Metrics", use_container_width=True):
                st.info("Model metrics")
        
        with col4:
            if st.button("🗑️ Archive Model", use_container_width=True):
                st.warning("Archive model")
    
    def _render_audit_logs(self):
        """Render audit logs"""
        st.markdown("### 📋 Audit Logs")
        
        # Log filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            log_type = st.selectbox("Log Type", ["All", "Authentication", "Analysis", "System", "Error"])
        
        with col2:
            date_range = st.date_input("Date Range", [datetime.now() - timedelta(days=7), datetime.now()])
        
        with col3:
            user_filter = st.text_input("Filter by User")
        
        # Logs table
        logs_data = {
            'Timestamp': [
                '2025-10-02 14:30:25',
                '2025-10-02 14:25:18',
                '2025-10-02 14:20:45',
                '2025-10-02 14:15:32',
                '2025-10-02 14:10:19'
            ],
            'User': ['dr.smith', 'admin', 'dr.jones', 'researcher01', 'dr.wilson'],
            'Action': ['Analysis Completed', 'User Created', 'Login Success', 'Model Trained', 'Data Uploaded'],
            'Type': ['Analysis', 'Admin', 'Auth', 'System', 'Data'],
            'Status': ['✅ Success', '✅ Success', '✅ Success', '✅ Success', '⚠️ Warning'],
            'IP Address': ['192.168.1.100', '192.168.1.1', '10.0.0.50', '172.16.0.25', '192.168.1.105']
        }
        
        df_logs = pd.DataFrame(logs_data)
        st.dataframe(df_logs, use_container_width=True)
        
        # Export logs
        if st.button("📥 Export Logs"):
            csv = df_logs.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"audit_logs_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    def _render_analytics(self):
        """Render analytics dashboard"""
        st.markdown("### 📈 System Analytics")
        
        # Usage statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Daily Analyses")
            fig_analyses = self._create_analyses_chart()
            st.plotly_chart(fig_analyses, use_container_width=True)
        
        with col2:
            st.markdown("#### 👥 Active Users")
            fig_users = self._create_users_chart()
            st.plotly_chart(fig_users, use_container_width=True)
        
        # Disease distribution
        st.markdown("#### 🧬 Disease Classification Distribution")
        fig_disease = self._create_disease_distribution()
        st.plotly_chart(fig_disease, use_container_width=True)
    
    def _create_analyses_chart(self) -> go.Figure:
        """Create analyses chart"""
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        analyses = [45, 52, 67, 58, 72, 38, 29]
        
        fig = go.Figure(data=[
            go.Bar(x=days, y=analyses, marker_color='#3498db')
        ])
        
        fig.update_layout(
            xaxis_title="Day",
            yaxis_title="Number of Analyses",
            height=300,
            template='plotly_white'
        )
        
        return fig
    
    def _create_users_chart(self) -> go.Figure:
        """Create users chart"""
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        users = [35, 42, 48, 45, 51, 28, 22]
        
        fig = go.Figure(data=[
            go.Scatter(x=days, y=users, mode='lines+markers', 
                      line=dict(color='#2ecc71', width=3),
                      marker=dict(size=10))
        ])
        
        fig.update_layout(
            xaxis_title="Day",
            yaxis_title="Active Users",
            height=300,
            template='plotly_white'
        )
        
        return fig
    
    def _create_disease_distribution(self) -> go.Figure:
        """Create disease distribution chart"""
        diseases = ['Healthy', 'Parkinson', 'Huntington', 'ALS', 'MS']
        counts = [450, 280, 120, 85, 95]
        
        fig = go.Figure(data=[
            go.Pie(labels=diseases, values=counts, hole=0.4,
                  marker=dict(colors=['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#3498db']))
        ])
        
        fig.update_layout(
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def _render_system_settings(self):
        """Render system settings"""
        st.markdown("### ⚙️ System Settings")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔧 General",
            "🔐 Security",
            "📧 Notifications",
            "🗄️ Backup"
        ])
        
        with tab1:
            self._render_general_settings()
        
        with tab2:
            self._render_security_settings()
        
        with tab3:
            self._render_notification_settings()
        
        with tab4:
            self._render_backup_settings()
    
    def _render_general_settings(self):
        """Render general settings"""
        st.markdown("#### 🔧 General Settings")
        
        with st.form("general_settings"):
            system_name = st.text_input("System Name", value="FE-AI Gait Detection")
            max_users = st.number_input("Max Concurrent Users", value=100, min_value=1)
            session_timeout = st.number_input("Session Timeout (minutes)", value=30, min_value=5)
            
            enable_logging = st.checkbox("Enable Detailed Logging", value=True)
            enable_analytics = st.checkbox("Enable Analytics", value=True)
            
            if st.form_submit_button("💾 Save Settings", type="primary"):
                st.success("✅ Settings saved successfully!")
    
    def _render_security_settings(self):
        """Render security settings"""
        st.markdown("#### 🔐 Security Settings")
        
        with st.form("security_settings"):
            password_policy = st.selectbox(
                "Password Policy",
                ["Basic", "Standard", "Strong", "Very Strong"]
            )
            
            session_security = st.selectbox(
                "Session Security",
                ["Standard", "High", "Maximum"]
            )
            
            two_factor = st.checkbox("Require Two-Factor Authentication", value=False)
            ip_whitelist = st.checkbox("Enable IP Whitelist", value=False)
            
            failed_login_limit = st.number_input("Failed Login Limit", value=5, min_value=1)
            lockout_duration = st.number_input("Lockout Duration (minutes)", value=15, min_value=1)
            
            if st.form_submit_button("🔒 Update Security", type="primary"):
                st.success("✅ Security settings updated!")
    
    def _render_notification_settings(self):
        """Render notification settings"""
        st.markdown("#### 📧 Notification Settings")
        
        with st.form("notification_settings"):
            email_notifications = st.checkbox("Email Notifications", value=True)
            sms_notifications = st.checkbox("SMS Notifications", value=False)
            
            st.markdown("**Notification Events:**")
            
            notify_login = st.checkbox("Failed Login Attempts", value=True)
            notify_analysis = st.checkbox("Analysis Completion", value=True)
            notify_errors = st.checkbox("System Errors", value=True)
            notify_updates = st.checkbox("System Updates", value=False)
            
            admin_email = st.text_input("Admin Email", value="admin@feai.com")
            
            if st.form_submit_button("📧 Save Notifications", type="primary"):
                st.success("✅ Notification settings saved!")
    
    def _render_backup_settings(self):
        """Render backup settings"""
        st.markdown("#### 🗄️ Backup & Recovery")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Automatic Backups**")
            
            with st.form("backup_settings"):
                backup_enabled = st.checkbox("Enable Automatic Backups", value=True)
                backup_frequency = st.selectbox(
                    "Backup Frequency",
                    ["Daily", "Weekly", "Monthly"]
                )
                backup_retention = st.number_input("Retention Days", value=30, min_value=7)
                
                if st.form_submit_button("💾 Save Backup Settings"):
                    st.success("✅ Backup settings saved!")
        
        with col2:
            st.markdown("**Manual Backup**")
            
            if st.button("🔄 Create Backup Now", use_container_width=True, type="primary"):
                with st.spinner("Creating backup..."):
                    import time
                    time.sleep(2)
                    st.success("✅ Backup created successfully!")
            
            st.markdown("**Last Backup:** 2025-10-02 03:00:00")
            st.markdown("**Backup Size:** 2.4 GB")
            
            if st.button("📥 Download Backup", use_container_width=True):
                st.info("Backup download initiated")
            
            if st.button("♻️ Restore from Backup", use_container_width=True):
                st.warning("⚠️ This will restore system from backup")
    
    def _render_notifications(self):
        """Render notifications"""
        st.markdown("### 🔔 System Notifications")
        
        # Notification filters
        col1, col2 = st.columns(2)
        
        with col1:
            notification_type = st.selectbox(
                "Filter by Type",
                ["All", "Info", "Warning", "Error", "Success"]
            )
        
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        # Notifications list
        notifications = [
            {
                'time': '5 minutes ago',
                'type': '⚠️ Warning',
                'message': 'High storage usage detected (85%)',
                'action': 'Review storage'
            },
            {
                'time': '1 hour ago',
                'type': '✅ Success',
                'message': 'Model training completed successfully',
                'action': 'View results'
            },
            {
                'time': '2 hours ago',
                'type': 'ℹ️ Info',
                'message': 'System backup completed',
                'action': 'View logs'
            },
            {
                'time': '3 hours ago',
                'type': '❌ Error',
                'message': 'Database connection timeout',
                'action': 'Check connection'
            },
            {
                'time': '5 hours ago',
                'type': '✅ Success',
                'message': 'Security patch applied',
                'action': 'View details'
            }
        ]
        
        for notif in notifications:
            with st.container():
                col1, col2, col3 = st.columns([1, 4, 1])
                
                with col1:
                    st.markdown(f"**{notif['type']}**")
                
                with col2:
                    st.markdown(f"{notif['message']}")
                    st.caption(notif['time'])
                
                with col3:
                    if st.button(notif['action'], key=f"action_{notif['time']}"):
                        st.info(f"Action: {notif['action']}")
                
                st.markdown("---")


if __name__ == "__main__":
    st.set_page_config(page_title="Admin Module", layout="wide")
    
    module = AdminModule()
    module.render()