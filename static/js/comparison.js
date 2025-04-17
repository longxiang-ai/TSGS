// 为所有cmp-container绑定鼠标进入、离开、移动事件以实现滑动对比效果
const cmpContainers = document.querySelectorAll('.cmp-container');
cmpContainers.forEach(container => {
    const slider = container.querySelector('.cmp-slider');
    let active = false // 当mouse移动到container上时，active为true此时move slider
    container.addEventListener('mouseenter', function(){
        active = true;
        slider.classList.add('sliding');

    });
    container.addEventListener('mouseleave', function(){
        active = false;
        slider.classList.remove('sliding');

    });
    container.addEventListener('mousemove', function(e){
        if(active){
            // 计算相对container的x坐标
            x = e.clientX - container.getBoundingClientRect().left;
            move(x);
        }
    });
    
    function move(x){
        x = Math.max(0, Math.min(x, container.offsetWidth)); // 限制x在container范围内,offsetWidth是元素的宽度不包括margin。
        container.querySelector('.top').style.width = x + 'px'; // slider图像
        slider.style.left = x - 15 + 'px'; // slider位置
    }
});
// let slider = document.querySelector('.cmp-slider');
// let container = document.querySelector('.cmp-container');

// 绑定按钮点击事件以切换图片
// function changeImages(cmpId, imgTopSrc, imgBottomSrc) {
//     const cmpContainer = document.getElementById(cmpId);
//     if (cmpContainer) {
//         const topImg = cmpContainer.querySelector('.top img');
//         const bottomImg = cmpContainer.querySelector('.bottom img');
        
//         if (topImg) {
//             topImg.src = imgTopSrc;
//         }
        
//         if (bottomImg) {
//             bottomImg.src = imgBottomSrc;
//         }
//     }
// }


// 预加载所有图片
document.addEventListener("DOMContentLoaded", function() {
    const images = [
        './static/images/cmp/2dgs/37_2dgs.png', './static/images/cmp/ours/37_ours.png',
        './static/images/cmp/2dgs/63_2dgs.png', './static/images/cmp/ours/63_ours.png',
        './static/images/cmp/2dgs/65_2dgs.png', './static/images/cmp/ours/65_ours.png',
        './static/images/cmp/2dgs/110_2dgs.png', './static/images/cmp/ours/110_ours.png',
        './static/images/cmp/2dgs/114_2dgs.png', './static/images/cmp/ours/114_ours.png',
        './static/images/cmp/2dgs/bonsai_2dgs.png', './static/images/cmp/ours/bonsai_ours.png',
        './static/images/cmp/2dgs/Caterpillar_2dgs.png', './static/images/cmp/ours/Caterpillar_ours.png',
        './static/images/cmp/2dgs/counter_2dgs.png', './static/images/cmp/ours/counter_ours.png'
    ];
    
    images.forEach(src => preloadImage(src));
    
    // 为每个比较容器设置初始高度
    const cmpContainers = document.querySelectorAll('.cmp-container');
    cmpContainers.forEach(container => {
        adjustContainerHeight(container);
    });
});

function preloadImage(src) {
    const img = new Image();
    img.src = src;
    img.onload = () => console.log(`${src} loaded successfully`);
    img.onerror = () => {
        console.error(`Failed to load ${src}, retrying...`);
        setTimeout(() => preloadImage(src), 1000); // Retry after 1 second
    };
}

function changeImages(event, cmpId, imgSrc1, imgSrc2) {
    const cmpContainer = document.getElementById(cmpId);
    if (!cmpContainer) return;
    
    const topImg = cmpContainer.querySelector('.top img');
    const bottomImg = cmpContainer.querySelector('.bottom img');
    if (!topImg || !bottomImg) return;
    
    // 获取容器当前宽度
    const containerWidth = cmpContainer.offsetWidth;
    
    // 创建新图像对象来获取图像尺寸
    const img = new Image();
    img.onload = function() {
        // 计算新的高度，保持图像宽高比
        const aspectRatio = img.height / img.width;
        const newHeight = containerWidth * aspectRatio;
        
        // 调整容器高度
        cmpContainer.style.height = newHeight + 'px';
        
        // 替换图像
        topImg.src = imgSrc1;
        bottomImg.src = imgSrc2;
    };
    img.onerror = function() {
        console.error('Failed to load image: ' + imgSrc1);
        // 即使加载失败也替换图像
        topImg.src = imgSrc1;
        bottomImg.src = imgSrc2;
    };
    img.src = imgSrc1; // 使用第一张图像来计算比例
    
    // 获取当前按钮的父元素容器
    const buttonContainer = event.target.parentElement;
    
    // 移除该容器内所有按钮的 .cmp-btn-checked 类
    const buttons = buttonContainer.querySelectorAll('.cmp-button');
    buttons.forEach(button => {
        button.classList.remove('cmp-btn-checked');
    });

    // 为当前点击的按钮添加 .cmp-btn-checked 类
    event.target.classList.add('cmp-btn-checked');
}

// 调整容器高度的函数
function adjustContainerHeight(container) {
    const img = container.querySelector('.bottom img') || container.querySelector('.top img');
    if (!img) return;
    
    const containerWidth = container.offsetWidth;
    
    // 如果图片已加载，直接调整高度
    if (img.complete) {
        const aspectRatio = img.naturalHeight / img.naturalWidth;
        container.style.height = (containerWidth * aspectRatio) + 'px';
    } else {
        // 如果图片未加载，等待加载完成后再调整
        img.onload = function() {
            const aspectRatio = img.naturalHeight / img.naturalWidth;
            container.style.height = (containerWidth * aspectRatio) + 'px';
        };
    }
}

