(function(){
  var root=document.documentElement,key='yewo-report-theme';
  try{var kept=localStorage.getItem(key); if(kept){root.setAttribute('data-theme',kept);}}catch(e){}
  var button=document.querySelector('button.theme');
  if(!button){return;}
  button.addEventListener('click',function(){
    var dark=matchMedia('(prefers-color-scheme: dark)').matches;
    var now=root.getAttribute('data-theme')||(dark?'dark':'light');
    var next=now==='dark'?'light':'dark';
    root.setAttribute('data-theme',next);
    try{localStorage.setItem(key,next);}catch(e){}
  });
})();
