document.getElementById('file').addEventListener('change', function(){
  if( this.value ){
    console.log( "Оппа, выбрали файл!" );
    console.log( this.value );
     
  } else { 
    console.log( "Файл не выбран" ); 
  }
  let Sylc = this.value;
});


async function send () {
  let val = Sylc;
  await eel.take_py(val)();
}


document.querySelector('button').addEventListener('click',function(){
  let file = document.getElementById('file').files[0];

   

    let reader = new FileReader();
    var allowedExtensions = /(.doc|.docx|.pdf|.txt)$/i;
    reader.readAsText(file);
    reader.onload = function(){
 
     let Sod = document.getElementById('Err')
     Sod.innerHTML = reader.result;
     Sod.innerHTML = Sylc;
 
     console.log(reader.result);
      
   ;}
   
   
 
  console.log(val);

})


