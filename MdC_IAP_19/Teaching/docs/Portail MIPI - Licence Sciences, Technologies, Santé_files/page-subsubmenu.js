// events onload
function addLoadEvent(func) {
  if (window.addEventListener)
    window.addEventListener("load", func, false);
  else if (window.attachEvent)
    window.attachEvent("onload", func);
}

// open and close lists
function aLists(IdList,id){
	var hasSecondLevelChild = false;
	id = (id !== undefined)?id:'dynamic_list';

	if (document.getElementsByTagName &&  document.getElementById && document.getElementById(id)) {
		var laList = document.getElementById(id);
		laList.className = 'dynamic_list';
		var li = laList.getElementsByTagName('LI');
   	for(var ii=0; ii<li.length-1; ++ii) {
			var subUl = li[ii].getElementsByTagName('UL');
			if (subUl[0]) {
				hasSecondLevelChild = true;
				subUl[0].id = IdList + ii;
				subUl[0].className = 'closed';

				var leLink = li[ii].getElementsByTagName('A')[0];

				iconLink = document.createElement('IMG');
				iconLink.src = imagesUriPrefix + 'img/sitemap/closed.gif';
				iconLink.alt = open_label;
				iconLink.className = "expand-image";

        leLink = document.createElement('a');
				leLink.href= "javascript:ShowHideList('" + IdList + ii + "')";
				leLink.className="icon";

        leLink.appendChild(iconLink);

				firstLink = li[ii].firstChild;
				li[ii].insertBefore(leLink,firstLink);
			}
		}

   		firstLink = laList.firstChild;
		if (hasSecondLevelChild)
		{
			leAllTxt = document.createTextNode(open_all_label);
			leAll = document.createElement('a');
			leAll.appendChild(leAllTxt);
			leAll.href= "javascript:ShowHideAll(\""+id+"\")";
			leAll.id= "all";
			leAll.className= "all_closed";
			laList.insertBefore(leAll,firstLink);
		}
	}
}
function ShowHideList(IdList){
	var subUl = document.getElementById(IdList);
	var leLink = subUl.parentNode.getElementsByTagName('A')[0];
	if (subUl.className == 'closed') {
		subUl.className = '';
		leLink.getElementsByTagName('IMG')[0].src = imagesUriPrefix + '/img/sitemap/opened.gif';
		leLink.getElementsByTagName('IMG')[0].alt = close_label;
	}
	else{
		subUl.className = 'closed';
		leLink.getElementsByTagName('IMG')[0].src = imagesUriPrefix + '/img/sitemap/closed.gif';
		leLink.getElementsByTagName('IMG')[0].alt =  open_label;
	}
}

function ShowHideAll(id){
	var laList = document.getElementById(id);
	var leAll = document.getElementById('all');
	var subUl = laList.getElementsByTagName('UL');
  if (leAll.className == 'all_closed') {
  	for(var ii=0; ii<= subUl.length-1; ++ii) {
  		subUl[ii].className = '';
			var leLink = subUl[ii].parentNode.getElementsByTagName('A')[0];
			leLink.getElementsByTagName('IMG')[0].src = imagesUriPrefix + '/img/sitemap/opened.gif';
			leLink.getElementsByTagName('IMG')[0].alt = close_label;
  	}
		leAll.className = 'all_opened';
		leTxt = leAll.firstChild;
		leTxt.nodeValue = close_all_label;
	}
  else {
  	for(var ii=0; ii<= subUl.length-1; ++ii) {
  		subUl[ii].className = 'closed';
			var leLink = subUl[ii].parentNode.getElementsByTagName('A')[0];
  		leLink.getElementsByTagName('IMG')[0].src = imagesUriPrefix + '/img/sitemap/closed.gif';
  		leLink.getElementsByTagName('IMG')[0].alt =  open_label;
		}
		leAll.className = 'all_closed';
		leTxt = leAll.firstChild;
		leTxt.nodeValue = open_all_label;
  }
}

function initializeList()
{
	$(".expand-image").css("position", "static");
}
//addLoadEvent(aLists);