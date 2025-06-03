import tkinter as tk
from tkinter import ttk, filedialog, messagebox
    
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import time
from ultralytics import YOLO

# 支持导入图片、导入视频、调用摄像头、左侧显示分割结果、右侧显示矫正后的图像

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("文档检测与校正系统")
        self.root.geometry("1200x800")
        
        # 加载模型
        model_path = "weights/best.pt"
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            messagebox.showwarning("警告", f"找不到模型文件: {model_path}\n请先训练模型或修改模型路径")
            self.model = None
        
        # 初始化变量
        self.cap = None  # 视频捕获对象
        self.is_camera_on = False
        self.current_image = None
        self.current_video_path = None
        self.video_thread = None
        self.stop_video = False
        self.available_cameras = self.get_available_cameras()  # 获取可用摄像头列表
        self.current_camera_index = 0  # 当前摄像头索引
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # 创建控制区域框架
        self.control_frame = ttk.LabelFrame(self.main_frame, text="控制面板")
        self.control_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # 创建图像显示区域框架
        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重，使显示区域可以扩展
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        # 左侧原始图像显示区域
        self.original_frame = ttk.LabelFrame(self.display_frame, text="原始图像 + 分割轮廓")
        self.original_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.original_canvas = tk.Canvas(self.original_frame, bg="black", width=500, height=500)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 右侧处理后图像显示区域
        self.processed_frame = ttk.LabelFrame(self.display_frame, text="矫正后图像")
        self.processed_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.processed_canvas = tk.Canvas(self.processed_frame, bg="black", width=500, height=500)
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 配置显示区域网格权重
        self.display_frame.columnconfigure(0, weight=1)
        self.display_frame.columnconfigure(1, weight=1)
        self.display_frame.rowconfigure(0, weight=1)
        
        # 添加控制按钮
        self.create_control_buttons()
        
        # 添加状态栏
        self.create_status_bar()
        
    def create_control_buttons(self):
        # 导入图片按钮
        self.import_image_btn = ttk.Button(self.control_frame, text="导入图片", command=self.import_image)
        self.import_image_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # 导入视频按钮
        self.import_video_btn = ttk.Button(self.control_frame, text="导入视频", command=self.import_video)
        self.import_video_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # 摄像头选择下拉框
        ttk.Label(self.control_frame, text="选择摄像头:").grid(row=0, column=2, padx=5, pady=5)
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(self.control_frame, textvariable=self.camera_var, state="readonly", width=10)
        self.camera_combo.grid(row=0, column=3, padx=5, pady=5)
        
        # 打开摄像头按钮
        self.camera_btn = ttk.Button(self.control_frame, text="打开摄像头", command=self.toggle_camera)
        self.camera_btn.grid(row=0, column=4, padx=5, pady=5)
        
        # 更新摄像头列表 - 移到这里，确保camera_btn已经创建
        self.update_camera_list()
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)
        
        # 停止按钮
        self.stop_btn = ttk.Button(self.control_frame, text="停止", command=self.stop_processing)
        self.stop_btn.grid(row=0, column=5, padx=5, pady=5)
        
        # 保存按钮
        self.save_btn = ttk.Button(self.control_frame, text="保存结果", command=self.save_result)
        self.save_btn.grid(row=0, column=6, padx=5, pady=5)
        
        # 导出按钮
        self.export_btn = ttk.Button(self.control_frame, text="导出结果", command=self.export_results)
        self.export_btn.grid(row=0, column=7, padx=5, pady=5)
        
        # 参数调节区域
        self.param_frame = ttk.LabelFrame(self.control_frame, text="参数调节")
        self.param_frame.grid(row=0, column=8, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # 置信度阈值
        ttk.Label(self.param_frame, text="置信度阈值:").grid(row=0, column=0, padx=5, pady=5)
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(self.param_frame, from_=0.1, to=1.0, 
                                   variable=self.confidence_var, orient=tk.HORIZONTAL, length=100)
        confidence_scale.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(self.param_frame, textvariable=self.confidence_var).grid(row=0, column=2, padx=5, pady=5)
    
    def create_status_bar(self):
        # 状态栏
        self.status_bar = ttk.Label(self.root, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.S))
    
    def import_image(self):
        """导入图片并处理"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                self.stop_processing()  # 停止任何正在进行的处理
                self.status_bar.config(text=f"正在处理图片: {os.path.basename(file_path)}")
                
                # 读取图片
                img = cv2.imread(file_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.current_image = img.copy()
                
                # 显示原始图片
                self.display_image(img_rgb, self.original_canvas)
                
                # 使用模型进行推理
                if self.model is not None:
                    self.process_image(img)
                else:
                    messagebox.showerror("错误", "模型未加载")
                
                self.status_bar.config(text=f"图片处理完成: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("错误", f"处理图片时出错: {str(e)}")
                self.status_bar.config(text="就绪")
    
    def import_video(self):
        """导入视频并处理"""
        file_path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if file_path:
            try:
                self.stop_processing()  # 停止任何正在进行的处理
                self.current_video_path = file_path
                self.status_bar.config(text=f"正在处理视频: {os.path.basename(file_path)}")
                
                # 创建并启动视频处理线程
                self.stop_video = False
                self.video_thread = threading.Thread(target=self.process_video)
                self.video_thread.daemon = True
                self.video_thread.start()
            except Exception as e:
                messagebox.showerror("错误", f"处理视频时出错: {str(e)}")
                self.status_bar.config(text="就绪")
    
    def toggle_camera(self):
        """打开/关闭摄像头"""
        if self.is_camera_on:
            self.stop_processing()
            self.camera_btn.config(text="打开摄像头")
            self.is_camera_on = False
        else:
            self.stop_processing()  # 停止任何正在进行的处理
            self.status_bar.config(text=f"正在使用摄像头 {self.current_camera_index}")
            self.camera_btn.config(text="关闭摄像头")
            
            # 创建并启动摄像头处理线程
            self.stop_video = False
            self.is_camera_on = True
            self.video_thread = threading.Thread(target=self.process_camera)
            self.video_thread.daemon = True
            self.video_thread.start()
    
    def stop_processing(self):
        """停止所有处理"""
        self.stop_video = True
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        self.is_camera_on = False
        self.camera_btn.config(text="打开摄像头")
        self.status_bar.config(text="就绪")
    
    def save_result(self):
        """保存处理结果"""
        if hasattr(self, 'warped_image') and self.warped_image is not None:
            file_path = filedialog.asksaveasfilename(
                title="保存结果",
                defaultextension=".jpg",
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("所有文件", "*.*")]
            )
            
            if file_path:
                try:
                    cv2.imwrite(file_path, self.warped_image)
                    self.status_bar.config(text=f"结果已保存到: {file_path}")
                except Exception as e:
                    messagebox.showerror("错误", f"保存结果时出错: {str(e)}")
        else:
            messagebox.showinfo("提示", "没有可保存的结果")
    
    def export_results(self):
        """导出左侧和右侧的图像结果"""
        if not hasattr(self, 'current_image') or self.current_image is None:
            messagebox.showinfo("提示", "没有可导出的图像")
            return
            
        # 选择保存目录
        save_dir = filedialog.askdirectory(title="选择保存目录")
        if not save_dir:
            return
            
        try:
            # 获取当前时间戳
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 保存原始图像（带分割轮廓）
            if hasattr(self, 'annotated_img'):
                original_path = os.path.join(save_dir, f"original_{timestamp}.jpg")
                cv2.imwrite(original_path, self.annotated_img)
            
            # 保存矫正后的图像
            if hasattr(self, 'warped_image') and self.warped_image is not None:
                warped_path = os.path.join(save_dir, f"corrected_{timestamp}.jpg")
                cv2.imwrite(warped_path, self.warped_image)
            
            self.status_bar.config(text=f"结果已导出到: {save_dir}")
            messagebox.showinfo("成功", "图像已成功导出")
            
        except Exception as e:
            messagebox.showerror("错误", f"导出结果时出错: {str(e)}")
    
    def process_image(self, img):
        """处理单张图片"""
        # 使用模型进行推理
        results = self.model(img, conf=self.confidence_var.get(), verbose=False)
        
        # 获取预测结果
        self.annotated_img = results[0].plot()  # 带分割掩码的注释图像
        
        # 显示带分割掩码的图像
        annotated_rgb = cv2.cvtColor(self.annotated_img, cv2.COLOR_BGR2RGB)
        self.display_image(annotated_rgb, self.original_canvas)
        
        # 执行文档校正
        self.perform_document_correction(img, results[0])
    
    def process_video(self):
        """处理视频"""
        self.cap = cv2.VideoCapture(self.current_video_path)
        
        while not self.stop_video and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 处理当前帧
            try:
                # 使用模型进行推理
                results = self.model(frame, conf=self.confidence_var.get(), verbose=False)
                
                # 获取预测结果
                annotated_frame = results[0].plot()  # 带分割掩码的注释图像
                
                # 显示带分割掩码的图像
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                self.display_image(annotated_rgb, self.original_canvas)
                
                # 执行文档校正
                self.perform_document_correction(frame, results[0])
                
            except Exception as e:
                print(f"处理视频帧时出错: {str(e)}")
            
            # 控制帧率
            time.sleep(0.03)  # 约30 FPS
        
        if self.cap is not None:
            self.cap.release()
        
        if not self.stop_video:  # 正常结束
            self.root.after(0, lambda: self.status_bar.config(text="视频处理完成"))
    
    def process_camera(self):
        """处理摄像头"""
        self.cap = cv2.VideoCapture(self.current_camera_index)  # 使用选定的摄像头
        
        if not self.cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("错误", f"无法打开摄像头 {self.current_camera_index}"))
            self.root.after(0, self.stop_processing)
            return
        
        while not self.stop_video and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 处理当前帧
            try:
                # 使用模型进行推理
                results = self.model(frame, conf=self.confidence_var.get(), verbose=False)
                
                # 获取预测结果
                if len(results) > 0 and len(results[0].boxes) > 0:
                    annotated_frame = results[0].plot()  # 带分割掩码的注释图像
                    
                    # 显示带分割掩码的图像
                    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    self.display_image(annotated_rgb, self.original_canvas)
                    
                    # 执行文档校正
                    self.perform_document_correction(frame, results[0])
                else:
                    # 如果没有检测到任何物体，只显示原始帧
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.display_image(rgb_frame, self.original_canvas)
                
            except Exception as e:
                print(f"处理摄像头帧时出错: {str(e)}")
                # 显示原始帧
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_image(rgb_frame, self.original_canvas)
            
            # 控制帧率
            time.sleep(0.03)  # 约30 FPS
        
        if self.cap is not None:
            self.cap.release()
    
    def perform_document_correction(self, img, result):
        """执行文档校正，基于分割掩码"""
        # 检查是否有检测结果
        if len(result.boxes) == 0 or not hasattr(result, 'masks') or result.masks is None:
            print("未检测到文档或分割掩码")
            return
        
        # 获取置信度最高的检测结果
        conf = result.boxes.conf.cpu().numpy()
        if len(conf) == 0:
            return
            
        max_conf_idx = np.argmax(conf)
        
        try:
            # 获取掩码数据
            mask = result.masks.data[max_conf_idx].cpu().numpy()
            
            # 将掩码转换为正确的形状
            if len(mask.shape) == 3:
                mask = mask.squeeze()  # 移除额外的维度
            
            # 转换为二值图像
            mask = (mask > 0.5).astype(np.uint8) * 255
            
            # 调整大小以匹配原始图像
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # 找到轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print("未找到有效轮廓")
                return
            
            # 找到最大的轮廓（假设是文档）
            max_contour = max(contours, key=cv2.contourArea)
            
            # 使用轮廓近似以找到文档的顶点
            epsilon = 0.02 * cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, epsilon, True)
            
            # 如果近似轮廓点太多，尝试找到四个最主要的点
            if len(approx) > 4:
                # 找到轮廓边界框
                rect = cv2.minAreaRect(max_contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                src_points = box.astype(np.float32)
            elif len(approx) == 4:
                # 如果正好是四个点，直接使用
                src_points = approx.reshape(4, 2).astype(np.float32)
            else:
                # 点数小于4，使用凸包
                hull = cv2.convexHull(max_contour)
                epsilon = 0.1 * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)
                
                if len(approx) < 4:
                    print(f"无法找到足够的角点，只有 {len(approx)} 个")
                    return
                
                # 保留前4个点
                src_points = approx[:4, 0, :].astype(np.float32)
            
            # 确保点是按顺序排列的
            # 首先按y坐标排序（从上到下）
            src_points = src_points[np.argsort(src_points[:, 1])]
            
            # 然后上面两个点按x坐标排序
            top_points = src_points[:2]
            top_points = top_points[np.argsort(top_points[:, 0])]
            
            # 下面两个点按x坐标排序
            bottom_points = src_points[2:]
            bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
            
            # 组合点（左上、右上、右下、左下）
            src_points = np.array([
                top_points[0],     # 左上
                top_points[1],     # 右上
                bottom_points[1],  # 右下
                bottom_points[0]   # 左下
            ], dtype=np.float32)
            
            # 计算目标文档的大小
            width_top = np.sqrt(((src_points[1][0] - src_points[0][0]) ** 2) + 
                               ((src_points[1][1] - src_points[0][1]) ** 2))
            width_bottom = np.sqrt(((src_points[2][0] - src_points[3][0]) ** 2) + 
                                  ((src_points[2][1] - src_points[3][1]) ** 2))
            max_width = max(int(width_top), int(width_bottom))
            
            height_left = np.sqrt(((src_points[3][0] - src_points[0][0]) ** 2) + 
                                 ((src_points[3][1] - src_points[0][1]) ** 2))
            height_right = np.sqrt(((src_points[2][0] - src_points[1][0]) ** 2) + 
                                  ((src_points[2][1] - src_points[1][1]) ** 2))
            max_height = max(int(height_left), int(height_right))
            
            # 设置目标点
            dst_points = np.array([
                [0, 0],                  # 左上
                [max_width - 1, 0],      # 右上
                [max_width - 1, max_height - 1],  # 右下
                [0, max_height - 1]      # 左下
            ], dtype=np.float32)
            
            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # 执行透视变换
            warped = cv2.warpPerspective(img, M, (max_width, max_height))
            self.warped_image = warped.copy()  # 保存校正后的图像
            
            # 显示校正后的图像
            warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            self.display_image(warped_rgb, self.processed_canvas)
        except Exception as e:
            print(f"执行文档校正时出错: {str(e)}")
    
    def display_image(self, img, canvas):
        """在Canvas上显示图像"""
        # 获取Canvas尺寸
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # 如果Canvas尚未配置好尺寸，使用默认尺寸
        if canvas_width <= 1:
            canvas_width = 500
        if canvas_height <= 1:
            canvas_height = 500
        
        # 调整图像大小以适应Canvas
        h, w = img.shape[:2]
        aspect_ratio = w / h
        
        if canvas_width / canvas_height > aspect_ratio:
            # Canvas更宽，以高度为基准
            new_height = canvas_height
            new_width = int(new_height * aspect_ratio)
        else:
            # Canvas更高，以宽度为基准
            new_width = canvas_width
            new_height = int(new_width / aspect_ratio)
        
        # 调整图像大小
        img_resized = cv2.resize(img, (new_width, new_height))
        
        # 将OpenCV图像转换为PIL格式
        img_pil = Image.fromarray(img_resized)
        
        # 将PIL图像转换为Tkinter PhotoImage
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # 在Canvas上显示图像
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, image=img_tk, anchor=tk.CENTER)
        
        # 保存对图像的引用以防止垃圾回收
        canvas.image = img_tk

    def get_available_cameras(self):
        """获取可用的摄像头设备列表"""
        available_cameras = []
        # 检查最多10个摄像头设备
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        return available_cameras
    
    def update_camera_list(self):
        """更新摄像头下拉列表"""
        camera_list = [f"摄像头 {i}" for i in self.available_cameras]
        if not camera_list:
            camera_list = ["无可用摄像头"]
            self.camera_btn.config(state=tk.DISABLED)
        else:
            self.camera_btn.config(state=tk.NORMAL)
        
        self.camera_combo['values'] = camera_list
        if camera_list and "无可用摄像头" not in camera_list:
            self.camera_combo.current(0)
            self.current_camera_index = self.available_cameras[0]
    
    def on_camera_selected(self, event=None):
        """处理摄像头选择变更事件"""
        if self.is_camera_on:
            # 如果摄像头已经开启，则切换到新选择的摄像头
            old_index = self.current_camera_index
            selected_index = self.camera_combo.current()
            if selected_index >= 0 and selected_index < len(self.available_cameras):
                self.current_camera_index = self.available_cameras[selected_index]
                
                if old_index != self.current_camera_index:
                    # 重新启动摄像头线程
                    self.stop_processing()
                    self.status_bar.config(text=f"切换到摄像头 {self.current_camera_index}")
                    self.toggle_camera()
        else:
            # 如果摄像头未开启，只更新索引
            selected_index = self.camera_combo.current()
            if selected_index >= 0 and selected_index < len(self.available_cameras):
                self.current_camera_index = self.available_cameras[selected_index]
            

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
