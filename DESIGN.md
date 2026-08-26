---
name: 声场本地配音工作台
description: 以电影声音剪辑台为母体的深色专业配音界面
colors:
  bench-black: "#090d14"
  studio-ink: "#0d131d"
  track-surface: "#16202d"
  rail-blue: "#243244"
  paper-white: "#f2f4f7"
  secondary-text: "#9aa8b9"
  quiet-text: "#6f7f92"
  playhead-amber: "#ffb45c"
  clone-blue: "#7db7d8"
  error-red: "#ff7d71"
  ready-green: "#7dd5a7"
typography:
  display:
    fontFamily: "-apple-system, BlinkMacSystemFont, SF Pro Text, PingFang SC, Noto Sans CJK SC, sans-serif"
    fontSize: "clamp(50px, 5.7vw, 86px)"
    fontWeight: 710
    lineHeight: 0.98
    letterSpacing: "-0.04em"
  title:
    fontFamily: "-apple-system, BlinkMacSystemFont, SF Pro Text, PingFang SC, Noto Sans CJK SC, sans-serif"
    fontSize: "20px"
    fontWeight: 680
    lineHeight: 1.2
    letterSpacing: "-0.02em"
  body:
    fontFamily: "-apple-system, BlinkMacSystemFont, SF Pro Text, PingFang SC, Noto Sans CJK SC, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "-apple-system, BlinkMacSystemFont, SF Pro Text, PingFang SC, Noto Sans CJK SC, sans-serif"
    fontSize: "10px"
    fontWeight: 640
    lineHeight: 1.4
    letterSpacing: "0.04em"
rounded:
  track: "5px"
  control: "8px"
  button: "10px"
  surface: "14px"
  dialog: "16px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "14px"
  lg: "22px"
  xl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.playhead-amber}"
    textColor: "{colors.bench-black}"
    rounded: "{rounded.button}"
    padding: "0 14px"
    height: "38px"
  button-quiet:
    backgroundColor: "{colors.track-surface}"
    textColor: "{colors.paper-white}"
    rounded: "{rounded.button}"
    padding: "0 14px"
    height: "38px"
  input:
    backgroundColor: "{colors.studio-ink}"
    textColor: "{colors.paper-white}"
    rounded: "{rounded.control}"
    padding: "9px 10px"
---

# Design System: 声场

## Overview

**Creative North Star: "电影声音剪辑台"**

声场把角色、台词和生成状态呈现为可剪接的声音素材，而不是普通管理后台。暗靛蓝工作面承载长时间操作，暖琥珀播放线只标记当前动作与生成焦点，克隆声线以冷蓝与场景造声区分。线框、轨道、标尺、场记编号和纸白对白条共同构成产品自己的操作语言。

系统是高密度但安静的：重要内容靠分栏、行高、字号和表面明度建立层级，不靠成排卡片或装饰性光效。欢迎页允许更大的构图与对白轨道示意，进入项目后则迅速收敛为项目片盘、对白剪辑台和角色声线架。

**Key Characteristics:**
- 暗色本地工作面，适合长时间音频编辑。
- 琥珀只服务主动作、播放线和场景造声识别。
- 冷蓝明确代表授权克隆及其参考素材。
- 数据密度来自轨道与分栏，而不是卡片堆叠。
- 状态同时使用文字、图形和颜色，不能只靠颜色表达。

## Colors

颜色来自暗室中的剪辑台、对白纸条与播放指示灯。

### Primary
- **播放线琥珀** (`#ffb45c`)：主按钮、当前播放线、场景造声图标与最需要注意的动作。

### Secondary
- **克隆声纹蓝** (`#7db7d8`)：授权克隆模式、参考音频与视频项目类型。
- **就绪绿** (`#7dd5a7`)：本地模型在线、生成通过和完成状态。
- **错误红** (`#ff7d71`)：失败、危险删除和明确恢复提示。

### Neutral
- **剪辑台黑** (`#090d14`)：顶栏、项目片盘和最深背景。
- **暗室靛墨** (`#0d131d`)：主工作区背景。
- **轨道表面** (`#16202d`)：按钮、选中项目和输入层次。
- **轨道分隔蓝** (`#243244`)：悬停、分段和更亮的结构层。
- **对白纸白** (`#f2f4f7`)：主文字、播放按钮和强调对白条。
- **次要文字** (`#9aa8b9`) 与 **静默文字** (`#6f7f92`)：说明、元数据和时间刻度。

**The Signal Color Rule.** 琥珀和克隆蓝必须绑定功能语义，不能作为随机装饰散落在页面上。

## Typography

**Display Font:** Apple 平台系统无衬线与苹方回退
**Body Font:** Apple 平台系统无衬线与苹方回退

**Character:** 字体像专业软件界面一样清楚、克制。首页标题可以强烈压缩字距并扩大尺度，编辑界面回到紧凑、稳定的工作字号；时间和编号使用等宽数字特性。

### Hierarchy
- **Display**（710，`clamp(50px, 5.7vw, 86px)`，0.98）：仅用于欢迎页产品命题。
- **Project Title**（680，20px，1.2）：项目名称和当前工作对象。
- **Section Title**（680，13px）：对白剪辑台、角色声线等工作区标题。
- **Body**（400，14px，1.55）：台词、表单和说明。
- **Label**（640，9–11px，适度字距）：轨道表头、模式、时间、来源和状态。

**The Dialogue Leads Rule.** 台词正文必须比控件标签和元数据明显更大、更亮，不能让配置项抢过正在表演的文字。

## Layout

桌面采用固定 64px 顶栏与 286px 项目片盘；工作区第一层是项目操作、状态、播放传输，第二层是可伸缩对白剪辑台与 360px 角色声线架。对白行使用 `TAKE / ROLE / DIALOGUE / CONTROL` 四列，编号和分隔线形成连续轨道，而非独立卡片。

1180px 以下收紧声线架；900px 以下将声线架移到对白之后；680px 以下项目片盘变为顶部横向片盘，对白行从四列重组为纵向可编辑段，避免依赖横向滚动。移动端保留项目主动作、角色、台词、表演和试听，不隐藏核心功能。

## Elevation & Depth

系统以色阶和 1px 结构线为主，阴影只用于真正浮在工作面之上的欢迎页轨道示意、侧滑项目设置、对话框和提示消息。静态工作区不使用每块内容都带边框加阴影的“幽灵卡片”。

### Shadow Vocabulary
- **浮动工作面** (`0 18px 46px rgba(0, 0, 0, .32)`)：对话框、侧面板、欢迎页大型示意和 toast。
- **对白条阴影** (`0 8px 20px rgba(0,0,0,.16)`)：只用于欢迎页的纸白声音条，不用于真实编辑行。

**The Flat Bench Rule.** 编辑工作面默认平坦；只有脱离文档流的浮层获得大阴影。

## Shapes

控制采用 8–10px 圆角，工作面和面板采用 14–16px 圆角；真正的对白轨道仅 5px 圆角，保持素材条而非卡片感。小状态与类型可以使用短圆角标签，药丸形只用于计数和极小状态。分割主要使用低对比度 1px 线。

## Components

### Buttons
- **Primary:** 琥珀底、深色字、10px 圆角、38px 高；用于创建、生成和确认。
- **Quiet:** 轨道表面底、纸白字、内嵌结构线；用于保存、导入和次级操作。
- **Icon:** 34px 方形点击区，透明静止，悬停时提升一个表面色阶。
- **Focus:** 2px 琥珀外轮廓，偏移 2px。

### Inputs / Fields
- 暗室靛墨底，8px 圆角，低对比度内嵌边线。
- 台词文本域在轨道内保持透明，让对白成为素材本身；设置与角色字段使用明确输入表面。
- 错误必须在附近以文字说明原因和恢复方式。

### Navigation
- 项目片盘以窄色槽区分脚本和视频；选中项目提升表面明度，不使用大面积品牌色。
- 角色标签以同样的窄色槽区分场景造声和授权克隆。

### Dialogue Strip
- 桌面是连续四列声音素材行；编号、角色、对白与控制共享一条基线。
- 手机端重组为角色、对白、表演、试听的纵向顺序，仍保留 Take 编号与移动控制。

### Voice Mode Switch
- 两段式控制只在角色声线架中使用。
- 场景造声选中态使用普通轨道表面；授权克隆选中态使用更冷的蓝色表面。
- 授权确认、参考音频和参考文本必须在克隆模式下可见，不能藏到额外设置页。

## Do's and Don'ts

### Do:
- **Do** 把角色、台词和声音来源放在首屏可操作区域。
- **Do** 用轨道、标尺、编号和分栏表达专业声音工作流。
- **Do** 保持造声与克隆在颜色、图标和文字上的三重区分。
- **Do** 在窄屏把表格重组为纵向台词编辑，而不是只要求横向滚动。
- **Do** 为生成、失败、授权和缓存状态提供清楚的文字反馈。

### Don't:
- **Don't** 将主工作区重构成同尺寸图标卡片网格。
- **Don't** 使用霓虹光晕、渐变文字、无功能玻璃效果或技术感网格装饰。
- **Don't** 用琥珀装饰无关内容；它必须意味着播放、创作或主动作。
- **Don't** 混淆授权克隆与虚构造声，也不要弱化授权确认。
- **Don't** 为每个静态容器同时使用边框和阴影。
