function jumptopage_scarlet(){
    var ind = document.getElementById('galind').value; //$('#galind');
    ind = parseInt(ind) //otherwise it is a string
    var fig_num = document.getElementsByClassName('row').length; // $('.row').length;
    var re = /^\d{1,10}$/;

    if (!re.test(ind))
    {
      alert('Index should be positive integers!');
      return false;
    } 
    else if (ind >= 300)
    {
      alert('Index should be less than 300!');
      return false;
    } 
    else {
      var page_num = Math.ceil((ind + 1) / fig_num)
      location.assign(`./page${page_num}.html#candy${ind}`)
      return false; // 不要提交表单
    }
  }

  function jumptopage_cutout(){
    var ind = document.getElementById('galind').value; //$('#galind');
    ind = parseInt(ind)
    var fig_num = 28;
    var re = /^\d{1,10}$/;

    if (!re.test(ind))
    {
      alert('Index should be positive integers!');
      return false;
    } 
    else if (ind >= 300)
    {
      alert('Index should be less than 300!');
      return false;
    } 
    else {
      var page_num = Math.ceil((ind + 1) / fig_num)
      location.assign(`./page${page_num}.html#candy${ind}`)
      return false; // 不要提交表单
    }
  }
