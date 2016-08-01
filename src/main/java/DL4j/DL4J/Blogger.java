package DL4j.DL4J;

import lombok.Getter; 
import lombok.Setter; 
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement; 

 

public class Blogger extends SequenceElement { 
    @Getter @Setter private int id; 
 
 
    public Blogger() { 

 
     } 
 
 
     public Blogger(int id) { 
         this.id = id; 
     } 
 
 
    @Override 
     public String getLabel() { 
         return null; 
     } 
 
 
    @Override 
     public String toJSON() { 
         return null; 
     } 
 } 