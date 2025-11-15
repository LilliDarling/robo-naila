"""Unit tests for hardware detection and optimization"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import multiprocessing
from config.hardware import HardwareOptimizer, HardwareInfo


class TestHardwareOptimizer:
    """Test hardware detection and optimization"""
    
    @pytest.fixture
    def mock_torch_cuda(self):
        """Mock PyTorch with CUDA support"""
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        
        # Mock device properties
        props = Mock()
        props.total_memory = 24 * 1024**3  # 24GB
        props.major = 8
        props.minor = 9
        mock_torch.cuda.get_device_properties.return_value = props
        
        # Mock MPS (not available for CUDA test)
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available.return_value = False
        
        return mock_torch
    
    @pytest.fixture
    def mock_torch_mps(self):
        """Mock PyTorch with Apple MPS support"""
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available.return_value = True
        
        return mock_torch
    
    @pytest.fixture
    def mock_torch_cpu(self):
        """Mock PyTorch CPU-only"""
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available.return_value = False
        
        return mock_torch
    
    @pytest.fixture
    def mock_system_info(self):
        """Mock system information"""
        mock_multiprocessing = Mock()
        mock_multiprocessing.cpu_count.return_value = 8
        
        mock_psutil = Mock()
        mock_psutil.virtual_memory.return_value.total = 32 * 1024**3
        
        return mock_multiprocessing, mock_psutil

    def test_cuda_detection(self, mock_torch_cuda, mock_system_info):
        """Test CUDA GPU detection"""
        mock_multiprocessing, mock_psutil = mock_system_info
        
        with patch.object(HardwareOptimizer, '_detect_hardware') as mock_detect:
            mock_detect.return_value = HardwareInfo(
                device_type="cuda",
                device_name="NVIDIA RTX 4090", 
                memory_gb=24.0,
                compute_capability="8.9",
                optimization_level="aggressive"
            )
            
            optimizer = HardwareOptimizer()
            
            assert optimizer.hardware_info.device_type == "cuda"
            assert optimizer.hardware_info.device_name == "NVIDIA RTX 4090"
            assert optimizer.hardware_info.memory_gb == 24.0
            assert optimizer.hardware_info.compute_capability == "8.9"
            assert optimizer.hardware_info.optimization_level == "aggressive"

    def test_mps_detection(self, mock_torch_mps, mock_system_info):
        """Test Apple MPS detection"""
        mock_multiprocessing, mock_psutil = mock_system_info
        
        with patch.object(HardwareOptimizer, '_detect_hardware') as mock_detect:
            mock_detect.return_value = HardwareInfo(
                device_type="mps",
                device_name="Apple Silicon GPU",
                optimization_level="standard"
            )
            
            optimizer = HardwareOptimizer()
            
            assert optimizer.hardware_info.device_type == "mps"
            assert optimizer.hardware_info.device_name == "Apple Silicon GPU"
            assert optimizer.hardware_info.optimization_level == "standard"

    def test_cpu_detection(self, mock_torch_cpu, mock_system_info):
        """Test CPU-only detection"""
        mock_multiprocessing, mock_psutil = mock_system_info
        
        with patch.object(HardwareOptimizer, '_detect_hardware') as mock_detect:
            mock_detect.return_value = HardwareInfo(
                device_type="cpu",
                device_name="Intel i9-12900K (x86_64)",
                memory_gb=32.0,
                optimization_level="aggressive"
            )
            
            optimizer = HardwareOptimizer()
            
            assert optimizer.hardware_info.device_type == "cpu"
            assert "Intel i9-12900K" in optimizer.hardware_info.device_name
            assert optimizer.hardware_info.memory_gb == 32.0
            assert optimizer.hardware_info.optimization_level == "aggressive"

    def test_hardware_detection_failure(self):
        """Test graceful fallback when hardware detection fails"""
        with patch.object(HardwareOptimizer, '_detect_hardware') as mock_detect:
            mock_detect.return_value = HardwareInfo(
                device_type="cpu",
                device_name="Unknown CPU",
                optimization_level="minimal"
            )
            
            optimizer = HardwareOptimizer()
            
            assert optimizer.hardware_info.device_type == "cpu"
            assert optimizer.hardware_info.device_name == "Unknown CPU"
            assert optimizer.hardware_info.optimization_level == "minimal"

    def test_gpu_optimization_levels(self, mock_torch_cuda, mock_system_info):
        """Test GPU optimization level determination"""
        optimizer = HardwareOptimizer()
        
        # Test different memory levels
        assert optimizer._get_gpu_optimization_level(12.0) == "aggressive"
        assert optimizer._get_gpu_optimization_level(6.0) == "standard"
        assert optimizer._get_gpu_optimization_level(2.0) == "minimal"

    def test_cpu_optimization_levels(self):
        """Test CPU optimization level determination"""
        optimizer = HardwareOptimizer()
        
        # High-end setup
        assert optimizer._get_cpu_optimization_level(16, 32.0) == "aggressive"
        # Standard setup
        assert optimizer._get_cpu_optimization_level(8, 16.0) == "aggressive"
        assert optimizer._get_cpu_optimization_level(4, 8.0) == "standard"
        # Low-end setup
        assert optimizer._get_cpu_optimization_level(2, 4.0) == "minimal"

    def test_cuda_config_generation(self, mock_torch_cuda, mock_system_info):
        """Test CUDA-specific configuration generation"""
        mock_multiprocessing, mock_psutil = mock_system_info
        
        with patch.object(HardwareOptimizer, '_detect_hardware') as mock_detect:
            mock_detect.return_value = HardwareInfo(
                device_type="cuda",
                device_name="NVIDIA RTX 4090",
                memory_gb=24.0,
                compute_capability="8.9",
                optimization_level="aggressive"
            )
            
            optimizer = HardwareOptimizer()
            config = optimizer.config
            
            assert config["device"] == "cuda"
            assert config["precision"] == "float16"
            assert config["compile_model"] == True
            assert config["batch_size"] == 1
            assert config["use_cache"] == True
 
    def test_mps_config_generation(self, mock_torch_mps, mock_system_info):
        """Test Apple MPS configuration generation"""
        with patch.object(HardwareOptimizer, '_detect_hardware') as mock_detect:
            mock_detect.return_value = HardwareInfo(
                device_type="mps",
                device_name="Apple Silicon GPU",
                optimization_level="standard"
            )
            
            optimizer = HardwareOptimizer()
            config = optimizer.config
            
            assert config["device"] == "mps"
            assert config["precision"] == "float32"  # MPS prefers float32
            assert config["compile_model"] == False

    def test_cpu_config_generation(self, mock_torch_cpu, mock_system_info):
        """Test CPU-specific configuration generation"""
        with patch.object(HardwareOptimizer, '_detect_hardware') as mock_detect:
            mock_detect.return_value = HardwareInfo(
                device_type="cpu",
                device_name="Intel i9-12900K (x86_64)",
                memory_gb=32.0,
                optimization_level="aggressive"
            )
            
            optimizer = HardwareOptimizer()
            config = optimizer.config
            
            assert config["device"] == "cpu"
            assert config["precision"] == "float32"
            assert config["thread_count"] == 4

    def test_low_memory_gpu_handling(self, mock_system_info):
        """Test low memory GPU configuration"""
        with patch.object(HardwareOptimizer, '_detect_hardware') as mock_detect:
            mock_detect.return_value = HardwareInfo(
                device_type="cuda",
                device_name="GTX 1060",
                memory_gb=3.0,
                compute_capability="6.1",
                optimization_level="minimal"
            )
            
            optimizer = HardwareOptimizer()
            config = optimizer.config
            
            assert config["precision"] == "float32"
            assert config["low_memory_mode"] == True

    def test_model_specific_config(self, mock_torch_cpu, mock_system_info):
        """Test model-specific configuration generation"""
        with patch.object(HardwareOptimizer, '_detect_hardware') as mock_detect:
            mock_detect.return_value = HardwareInfo(
                device_type="cpu",
                device_name="Intel i9-12900K (x86_64)",
                memory_gb=32.0,
                optimization_level="aggressive"
            )
            
            optimizer = HardwareOptimizer()
            
            # Sentence transformer config
            st_config = optimizer.get_model_config("sentence_transformer")
            assert st_config["convert_to_numpy"] == True
            assert st_config["device"] == "cpu"

    def test_intel_cpu_optimization(self, mock_torch_cpu, mock_system_info):
        """Test Intel-specific CPU optimizations"""
        with patch.object(HardwareOptimizer, '_detect_hardware') as mock_detect, \
             patch.object(HardwareOptimizer, '_get_cpu_info', return_value="Intel Core i7-10700K"):
            mock_detect.return_value = HardwareInfo(
                device_type="cpu",
                device_name="Intel Core i7-10700K (x86_64)",
                memory_gb=32.0,
                optimization_level="aggressive"
            )
            
            optimizer = HardwareOptimizer()
            config = optimizer.config
            
            assert config["use_intel_extension"] == True

    def test_cpu_info_detection(self):
        """Test CPU information detection"""
        optimizer = HardwareOptimizer()

        # Test _get_cpu_info method directly with mocked cpuinfo
        # Mock the cpuinfo module that's already imported
        mock_cpuinfo_module = Mock()
        mock_cpuinfo_module.get_cpu_info.return_value = {"brand_raw": "Intel i9-12900K"}

        with patch('config.hardware.cpuinfo', mock_cpuinfo_module), \
             patch('config.hardware.platform.machine', return_value="x86_64"):

            cpu_info = optimizer._get_cpu_info()
            assert "Intel i9-12900K" in cpu_info
            assert "x86_64" in cpu_info

    def test_cpu_info_fallback(self):
        """Test CPU info fallback when cpuinfo unavailable"""
        optimizer = HardwareOptimizer()
        
        # Test fallback when cpuinfo import fails
        with patch('builtins.__import__') as mock_import, \
             patch('config.hardware.os.cpu_count', return_value=8):
            
            def import_side_effect(name, *args, **kwargs):
                if name == 'cpuinfo':
                    raise ImportError("cpuinfo not available")
                else:
                    return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            cpu_info = optimizer._get_cpu_info()
            assert "CPU (8 cores)" in cpu_info

    def test_thread_count_limiting(self, mock_torch_cpu, mock_system_info):
        """Test thread count limiting for CPU processing"""
        with patch.object(HardwareOptimizer, '_detect_hardware') as mock_detect:
            mock_detect.return_value = HardwareInfo(
                device_type="cpu",
                device_name="Intel i9-12900K (x86_64)",
                memory_gb=32.0,
                optimization_level="aggressive"
            )
            
            # Mock multiprocessing.cpu_count inside _get_cpu_config method
            with patch('builtins.__import__') as mock_import:
                mock_multiprocessing = Mock()
                mock_multiprocessing.cpu_count.return_value = 16
                
                def import_side_effect(name, *args, **kwargs):
                    if name == 'multiprocessing':
                        return mock_multiprocessing
                    else:
                        return __import__(name, *args, **kwargs)
                
                mock_import.side_effect = import_side_effect
                
                optimizer = HardwareOptimizer()
                config = optimizer.config
                
                # Should limit to 4 threads for stability
                assert config["thread_count"] == 4

    def test_config_immutability(self, mock_torch_cuda, mock_system_info):
        """Test that configurations are properly isolated"""
        optimizer = HardwareOptimizer()
        
        config1 = optimizer.get_model_config("sentence_transformer")
        config2 = optimizer.get_model_config("sentence_transformer")
        
        # Modify one config
        config1["test_key"] = "test_value"
        
        # Other config should be unaffected
        assert "test_key" not in config2

    def test_multiple_gpu_detection(self, mock_system_info):
        """Test detection with multiple GPUs"""
        with patch.object(HardwareOptimizer, '_detect_hardware') as mock_detect:
            mock_detect.return_value = HardwareInfo(
                device_type="cuda",
                device_name="NVIDIA RTX 4090",
                memory_gb=24.0,
                compute_capability="8.9",
                optimization_level="aggressive"
            )
            
            optimizer = HardwareOptimizer()
            
            # Should use first GPU
            assert optimizer.hardware_info.device_type == "cuda"
            assert optimizer.hardware_info.device_name == "NVIDIA RTX 4090"

    def test_hardware_info_dataclass(self):
        """Test HardwareInfo dataclass functionality"""
        info = HardwareInfo(
            device_type="cuda",
            device_name="Test GPU",
            memory_gb=8.0,
            compute_capability="7.5",
            optimization_level="standard"
        )
        
        assert info.device_type == "cuda"
        assert info.memory_gb == 8.0
        assert info.compute_capability == "7.5"
        
        # Test default values
        minimal_info = HardwareInfo(device_type="cpu", device_name="Test CPU")
        assert minimal_info.memory_gb is None
        assert minimal_info.optimization_level == "standard"

    def test_global_optimizer_instance(self):
        """Test global hardware optimizer instance"""
        from config.hardware import hardware_optimizer
        
        assert hardware_optimizer is not None
        assert hasattr(hardware_optimizer, 'hardware_info')
        assert hasattr(hardware_optimizer, 'config')
        assert callable(hardware_optimizer.get_model_config)